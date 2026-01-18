# system/loader.py
import os
import sys
import json
import torch
import importlib.util
from omegaconf import OmegaConf

from MLsystem.registry import Registry
from MLsystem.inspector import find_config_class
from MLsystem.checkpoint_manager import CheckpointManager
from MLsystem.utils.env_manager import EnvManager


class ExperimentLoader:
    """
    実験ID(Hash)に基づき、設定・モデル・データセット・チェックポイントを復元するクラス。
    """

    def __init__(self, hash_id):
        self.hash_id = hash_id
        self.exp_dir = os.path.join(EnvManager().output_dir, "experiments", hash_id)
        self.config_path = os.path.join(self.exp_dir, "config_diff.json")

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found for experiment {hash_id}")

        with open(self.config_path, "r") as f:
            self.diff_payload = json.load(f)

        self.model_name = self.diff_payload["model"]
        self.adapter_name = self.diff_payload.get("adapter")
        self.dataset_name = self.diff_payload["dataset"]

        self.registry = Registry()
        self.overrides = {}

    def update_overrides(self, overrides):
        """外部からランタイムオーバーライド（修正用パラメータなど）を設定する"""
        if overrides:
            self.overrides.update(overrides)

    def _restore_params(self, config_cls, diff):
        """Configクラスのデフォルト値にDiffを適用してパラメータを復元"""
        # Configクラスそのものを受け取るように変更
        if not config_cls:
            # 見つからない場合はdiffをそのまま返す（フォールバック）
            return diff or {}

        # デフォルト設定を作成 (OmegaConf)
        conf = OmegaConf.structured(config_cls)

        # 差分をマージ
        if diff:
            conf = OmegaConf.merge(conf, diff)

        # 辞書に戻して返す
        return OmegaConf.to_container(conf, resolve=True)

    def load_modules(self, overrides=None):
        """定義モジュールとパラメータをロードして返す（インスタンス化はしない）"""
        if overrides is None:
            overrides = self.overrides
        # overrides: {"model_diff": {...}, "data_diff": {...}, ...}
        try:
            # クラス/モジュール読み込み (Registry)
            ModelClass = self.registry.get_main_class("models", self.model_name)
            adapter_mod = self.registry.get_adapter_module(
                self.model_name, self.adapter_name
            )
            DataModuleClass = self.registry.get_main_class(
                "datasets", self.dataset_name
            )

            # Configクラス読み込み
            model_cls = self.registry.get_config_class("models", self.model_name)
            adapter_cls = self.registry.get_config_class(
                "models", self.model_name, self.adapter_name
            )
            data_cls = self.registry.get_config_class("datasets", self.dataset_name)

            # 差分取得（オーバーライド適用）
            def get_diff(key):
                diff = self.diff_payload.get(key, {}).copy()
                if overrides and key in overrides:
                    diff.update(overrides[key])
                return diff

            # パラメータ復元
            model_params = self._restore_params(model_cls, get_diff("model_diff"))
            adapter_params = self._restore_params(
                adapter_cls, get_diff("adapter_diff")
            )
            data_params = self._restore_params(data_cls, get_diff("data_diff"))

            return {
                "ModelClass": ModelClass,
                "adapter_mod": adapter_mod,
                "DataModuleClass": DataModuleClass,
                "model_params": model_params,
                "adapter_params": adapter_params,
                "data_params": data_params,
            }
        except ImportError as e:
            raise ImportError(f"Failed to load definition modules: {e}")

    def setup(self, stage="test", overrides=None):
        """
        DataModuleとModelを初期化し、Checkpointがあればロードした状態で返す。
        return: (model, datamodule)
        """
        if overrides is None:
            overrides = self.overrides
        modules = self.load_modules(overrides=overrides)

        # 1. AdapterからTransform取得
        input_transform = modules["adapter_mod"].get_input_transform(
            modules["adapter_params"]
        )

        # 2. DataModule作成
        # diff_payloadから共通設定も復元する
        common_conf_mod = EnvManager().get_common_config_module()
        common_config_cls = find_config_class(common_conf_mod)

        common_diff = self.diff_payload.get("common_diff", {}).copy()
        if overrides and "common_diff" in overrides:
            common_diff.update(overrides["common_diff"])

        common_params = self._restore_params(common_config_cls, common_diff)

        # パラメータ結合ロジックを execute_train.py と統一 (Common < Dataset < Adapter < Model)
        all_params = {}
        all_params.update(common_params)
        all_params.update(modules["data_params"])
        all_params.update(modules["adapter_params"])
        all_params.update(modules["model_params"])

        # DataModuleクラスを使用
        DataModuleClass = modules["DataModuleClass"]
        datamodule = DataModuleClass(adapter_transform=input_transform, **all_params)

        datamodule.prepare_data()
        datamodule.setup(stage=stage)

        # 3. Model作成
        data_meta = {
            k: getattr(datamodule, k) for k in dir(datamodule) if not k.startswith("_")
        }
        model_init_args = modules["adapter_mod"].get_model_init_args(
            data_meta, modules["adapter_params"]
        )

        # Model引数の結合 (all_params + model_init_args)
        final_model_kwargs = {**all_params, **model_init_args}

        # Modelクラスを使用
        ModelClass = modules["ModelClass"]
        model = ModelClass(**final_model_kwargs)

        self.model = model
        self.datamodule = datamodule

        return model, datamodule

    def load_model_from_checkpoint(self, ckpt_path):
        """指定パスのCheckpointをロードしたモデルを返す"""
        model, _ = self.setup(stage="test")
        print(f"Loading weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def get_checkpoints(self):
        """
        利用可能なチェックポイントの一覧を取得する。
        """
        manager = CheckpointManager(self.exp_dir)
        return manager.list_checkpoints()

    def get_checkpoint_path(self, epoch=None):
        """
        指定エポックのCKPTパスを返す。
        epoch=Noneの場合は 'last.ckpt' または 最もエポック数が大きいものを返す。
        """
        manager = CheckpointManager(self.exp_dir)
        return manager.get_path(epoch)
