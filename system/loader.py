# system/loader.py
import os
import json
import torch
import importlib
import glob
import re

class ExperimentLoader:
    """
    実験ID(Hash)に基づき、設定・モデル・データセット・チェックポイントを復元するクラス。
    """
    def __init__(self, hash_id):
        self.hash_id = hash_id
        self.exp_dir = os.path.join("output", "experiments", hash_id)
        self.config_path = os.path.join(self.exp_dir, "config_diff.json")
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found for experiment {hash_id}")
            
        with open(self.config_path, "r") as f:
            self.diff_payload = json.load(f)
            
        self.model_name = self.diff_payload["model"]
        self.adapter_name = self.diff_payload.get("adapter")
        self.dataset_name = self.diff_payload["dataset"]

    def _restore_params(self, schema, diff):
        """スキーマのデフォルト値にDiffを適用してパラメータを復元"""
        params = {k: v["default"] for k, v in schema.items()}
        # ui_mode="hidden"などでdefaultが含まれない場合などのケアが必要ならここで行う
        if diff:
            params.update(diff)
        return params

    def load_modules(self):
        """定義モジュールとパラメータをロードして返す（インスタンス化はしない）"""
        try:
            # モジュール読み込み
            model_mod = importlib.import_module(f"definitions.models.{self.model_name}.model")
            adapter_mod = importlib.import_module(f"definitions.models.{self.model_name}.adapters.{self.adapter_name}.adapter")
            data_mod = importlib.import_module(f"definitions.datasets.{self.dataset_name}.datamodule")
            
            # スキーマ読み込み
            model_conf_mod = importlib.import_module(f"definitions.models.{self.model_name}.config")
            adapter_conf_mod = importlib.import_module(f"definitions.models.{self.model_name}.adapters.{self.adapter_name}.config")
            data_conf_mod = importlib.import_module(f"definitions.datasets.{self.dataset_name}.config")
            
            # パラメータ復元
            model_params = self._restore_params(model_conf_mod.CONFIG_SCHEMA, self.diff_payload.get("model_diff", {}))
            adapter_params = self._restore_params(adapter_conf_mod.CONFIG_SCHEMA, self.diff_payload.get("adapter_diff", {}))
            data_params = self._restore_params(data_conf_mod.CONFIG_SCHEMA, self.diff_payload.get("data_diff", {}))
            
            return {
                "model_mod": model_mod,
                "adapter_mod": adapter_mod,
                "data_mod": data_mod,
                "model_params": model_params,
                "adapter_params": adapter_params,
                "data_params": data_params
            }
        except ImportError as e:
            raise ImportError(f"Failed to load definition modules: {e}")

    def setup(self, stage="test"):
        """
        DataModuleとModelを初期化し、Checkpointがあればロードした状態で返す。
        return: (model, datamodule)
        """
        modules = self.load_modules()
        
        # 1. AdapterからTransform取得
        input_transform = modules["adapter_mod"].get_input_transform(modules["adapter_params"])
        
        # 2. DataModule作成
        # data_paramsだけでなく、common_params (batch_size等) も渡す必要がある
        # diff_payloadから共通設定も復元する
        import common.config as common_conf_mod
        common_params = self._restore_params(common_conf_mod.CONFIG_SCHEMA, self.diff_payload.get("common_diff", {}))
        
        # パラメータを結合 (data_paramsが優先)
        combined_data_params = {**common_params, **modules["data_params"]}
        
        datamodule = modules["data_mod"].create_datamodule(combined_data_params, adapter_transform=input_transform)
        datamodule.prepare_data()
        datamodule.setup(stage=stage)
        
        # 3. Model作成
        data_meta = {k: getattr(datamodule, k) for k in dir(datamodule) if not k.startswith('_')}
        model_init_args = modules["adapter_mod"].get_model_init_args(data_meta, modules["adapter_params"])
        final_model_kwargs = {**modules["model_params"], **model_init_args}
        
        model = modules["model_mod"].Model(**final_model_kwargs)
        
        return model, datamodule

    def load_model_from_checkpoint(self, ckpt_path):
        """指定パスのCheckpointをロードしたモデルを返す"""
        model, _ = self.setup(stage="test")
        print(f"Loading weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def get_checkpoints(self):
        """
        利用可能なチェックポイントの一覧を取得する。
        return: List of dict {'epoch': int, 'path': str, 'type': 'epoch'|'last'}
        """
        # checkpointディレクトリを探す
        # 構造: output/experiments/{hash}/lightning_logs/version_*/checkpoints/*.ckpt
        # または output/experiments/{hash}/lightning_logs/checkpoints/*.ckpt (versionなしの場合)
        
        log_dir = os.path.join(self.exp_dir, "lightning_logs")
        if not os.path.exists(log_dir):
            return []
            
        candidate_files = []
        
        # 再帰的に .ckpt を探す
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.endswith(".ckpt"):
                    candidate_files.append(os.path.join(root, file))
        
        results = []
        for path in candidate_files:
            fname = os.path.basename(path)
            info = {"path": path, "type": "unknown", "epoch": -1}
            
            if fname == "last.ckpt":
                info["type"] = "last"
                # last.ckptの中身を見ないとepochは不明だが、ここでは一旦区別する
            else:
                # epoch=X-step=Y.ckpt の形式をパース
                m = re.search(r"epoch=(\d+)", fname)
                if m:
                    info["epoch"] = int(m.group(1))
                    info["type"] = "epoch"
            
            results.append(info)
            
        # エポック順にソート (lastは除外または末尾へ)
        # ここでは epoch が判明しているものを優先
        results.sort(key=lambda x: x["epoch"])
        return results

    def get_checkpoint_path(self, epoch=None):
        """
        指定エポックのCKPTパスを返す。
        epoch=Noneの場合は 'last.ckpt' または 最もエポック数が大きいものを返す。
        """
        ckpts = self.get_checkpoints()
        if not ckpts:
            return None
            
        if epoch is not None:
            # 指定エポックを探す
            for c in ckpts:
                if c["epoch"] == epoch:
                    return c["path"]
            return None
        else:
            # 最新を探す
            # 'last.ckpt' があればそれを優先
            last = next((c for c in ckpts if c["type"] == "last"), None)
            if last:
                return last["path"]
            # なければエポック最大のもの
            return ckpts[-1]["path"]