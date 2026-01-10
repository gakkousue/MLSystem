# system/train.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import importlib
import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from system.hashing import compute_combined_hash

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    print(f"Loaded Config: Model={cfg.model}, Adapter={cfg.adapter}, Dataset={cfg.dataset}")

    # 1. モジュール読み込み
    try:
        # ロジック用モジュール
        model_mod = importlib.import_module(f"definitions.models.{cfg.model}.model")
        # Adapterのロード (definitions.models.{model}.adapters.{adapter})
        adapter_mod = importlib.import_module(f"definitions.models.{cfg.model}.adapters.{cfg.adapter}.adapter")
        data_mod  = importlib.import_module(f"definitions.datasets.{cfg.dataset}.datamodule")
        
        # Plot用モジュール (存在すればロード)
        try:
            data_plot_mod = importlib.import_module(f"definitions.datasets.{cfg.dataset}.plots")
        except ImportError: data_plot_mod = None
        
        try:
            adapter_plot_mod = importlib.import_module(f"definitions.models.{cfg.model}.adapters.{cfg.adapter}.plot")
        except ImportError: adapter_plot_mod = None

        # スキーマ用モジュール (config.py)
        model_conf_mod = importlib.import_module(f"definitions.models.{cfg.model}.config")
        adapter_conf_mod = importlib.import_module(f"definitions.models.{cfg.model}.adapters.{cfg.adapter}.config")
        data_conf_mod  = importlib.import_module(f"definitions.datasets.{cfg.dataset}.config")
    except ImportError as e:
        print(f"Error: Modules not found. {e}")
        return

    # 2. ユーザー設定の展開
    user_common_params  = OmegaConf.to_container(cfg.common, resolve=True)         if "common" in cfg else {}
    user_model_params   = OmegaConf.to_container(cfg.model_params, resolve=True)   if "model_params" in cfg else {}
    user_adapter_params = OmegaConf.to_container(cfg.adapter_params, resolve=True) if "adapter_params" in cfg else {}
    user_data_params    = OmegaConf.to_container(cfg.data_params, resolve=True)    if "data_params" in cfg else {}
    
    user_model_params["_name"]   = cfg.model
    user_adapter_params["_name"] = cfg.adapter
    user_data_params["_name"]    = cfg.dataset
    
    # 共通設定スキーマ
    import common.config as common_conf_mod

    # 3. ハッシュ計算 (Adapterも含む)
    hash_id, diff_payload = compute_combined_hash(
        common_conf_mod.CONFIG_SCHEMA,   user_common_params,
        model_conf_mod.CONFIG_SCHEMA,    user_model_params,
        adapter_conf_mod.CONFIG_SCHEMA,  user_adapter_params,
        data_conf_mod.CONFIG_SCHEMA,     user_data_params
    )

    save_dir = os.path.join("output", "experiments", hash_id)
    print(f"Experiment Hash ID: {hash_id}")

    if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "done")):
        print(">> Experiment already finished. Skipping.")
        return
    
    os.makedirs(save_dir, exist_ok=True)

    # 4. DataModuleの準備
    # Adapterから変換関数(transform)を取得してDatasetに渡す
    input_transform = adapter_mod.get_input_transform(user_adapter_params)
    
    # common設定(batch_size等)をデータ設定にマージして渡す
    combined_data_params = {**user_common_params, **user_data_params}
    
    datamodule = data_mod.create_datamodule(combined_data_params, adapter_transform=input_transform)
    
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # --- Plot Phase 1: 生データの可視化 ---
    if data_plot_mod and hasattr(data_plot_mod, "plot_samples"):
        try:
            data_plot_mod.plot_samples(datamodule, save_dir)
        except Exception as e:
            print(f"Warning: Plot Phase 1 failed: {e}")

    # --- Plot Phase 2: Adapter変換後データの可視化 ---
    if adapter_plot_mod and hasattr(adapter_plot_mod, "plot_transformed_sample"):
        try:
            # Datasetから1サンプル取得 (Augmentation + Adapter変換済み)
            # train_dataloaderのdatasetはSubsetの可能性があるためアクセスに注意
            dataset_inst = datamodule.train_dataloader().dataset
            if len(dataset_inst) > 0:
                sample_data = dataset_inst[0] # (img_tensor, label)
                adapter_plot_mod.plot_transformed_sample(sample_data, save_dir)
        except Exception as e:
            print(f"Warning: Plot Phase 2 failed: {e}")

    # 5. Modelの準備
    # Datasetのメタ情報を取得し、Adapterを通してModel用引数に変換する
    data_meta = {k: getattr(datamodule, k) for k in dir(datamodule) if not k.startswith('_')}
    
    # Adapter: "このデータなら、Modelにはこういう引数(in_channels=1など)を渡してね"
    model_init_args = adapter_mod.get_model_init_args(data_meta, user_adapter_params)
    
    # ユーザー設定とマージ (ユーザー設定が優先だが、構造的な引数はAdapter主導)
    final_model_kwargs = {**user_model_params, **model_init_args}
    
    model = model_mod.Model(**final_model_kwargs)

    # 6. 設定の保存
    with open(os.path.join(save_dir, "config_diff.json"), "w") as f:
        json.dump(diff_payload, f, indent=4)

    # 8. ロガーと再開設定
    # version="" にすることで version_0 などのフォルダ自動生成を停止し、常に同じ場所を使う
    logger = pl.loggers.TensorBoardLogger(
        save_dir=save_dir,
        name="lightning_logs",
        version="" 
    )

    # 過去のチェックポイントがあれば自動的に再開(Resume)するロジック
    ckpt_path = None
    # Loggerの設定により、チェックポイントは lightning_logs/checkpoints に保存される
    ckpt_dir = os.path.join(save_dir, "lightning_logs", "checkpoints")
    
    if os.path.exists(ckpt_dir):
        # ファイルリストを取得
        ckpts = [c for c in os.listdir(ckpt_dir) if c.endswith(".ckpt")]
        if ckpts:
            # 'last.ckpt' があればそれを最優先、なければ名前順で最後（最新エポック）のものを使う
            if "last.ckpt" in ckpts:
                ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
            else:
                ckpts.sort()
                ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
            
            print(f">> Found checkpoint. Resuming from: {ckpt_path}")

    # 9. 学習実行
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        logger=logger,  # バージョン固定のロガーを使用
        max_epochs=user_model_params.get("epochs", 5),
        accelerator="auto",
        devices=1
    )
    
    # ckpt_path引数を渡すことで、前回の中断地点から学習を再開できる
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    # 完了フラグ
    with open(os.path.join(save_dir, "done"), "w") as f:
        f.write("finished")

if __name__ == "__main__":
    main()