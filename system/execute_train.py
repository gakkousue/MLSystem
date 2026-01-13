# system/execute_train.py
import os
import sys

# プロジェクトルートにパスを通す
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import importlib
import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from system.hashing import compute_combined_hash
from system.registry import Registry
from system.inspector import find_config_class # common設定用に残す
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint  # 新規追加

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    print(f"Loaded Config: Model={cfg.model}, Adapter={cfg.adapter}, Dataset={cfg.dataset}")

    registry = Registry()

    # 1. モジュール読み込み (Registry使用)
    try:
        # クラス定義の取得
        # Model
        ModelClass = registry.get_main_class("models", cfg.model)
        # DataModule
        DataModuleClass = registry.get_main_class("datasets", cfg.dataset)
        # Adapterモジュール (関数群)
        adapter_mod = registry.get_adapter_module(cfg.model, cfg.adapter)
        
        # Configクラス定義の取得
        model_config_cls = registry.get_config_class("models", cfg.model)
        adapter_config_cls = registry.get_config_class("models", cfg.model, cfg.adapter)
        data_config_cls = registry.get_config_class("datasets", cfg.dataset)
        
    except Exception as e:
        print(f"Error loading modules from registry: {e}")
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
    
    # Common設定は従来通り
    common_cls = find_config_class(common_conf_mod)
    
    # 他はRegistryから取得済み
    model_cls = model_config_cls
    adapter_cls = adapter_config_cls
    data_cls = data_config_cls

    # 3. ハッシュ計算 (Adapterも含む)
    hash_id, diff_payload = compute_combined_hash(
        common_cls,   user_common_params,
        model_cls,    user_model_params,
        adapter_cls,  user_adapter_params,
        data_cls,     user_data_params
    )

    save_dir = os.path.join("output", "experiments", hash_id)
    print(f"Experiment Hash ID: {hash_id}")

    # if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "done")):
    #     print(">> Experiment already finished. Skipping.")
    #     return
    
    os.makedirs(save_dir, exist_ok=True)

    # 4. DataModuleの準備
    # Adapterから変換関数(transform)を取得してDatasetに渡す
    input_transform = adapter_mod.get_input_transform(user_adapter_params)
    
    # 全パラメータを統合する (Common < Dataset < Adapter < Model の順で上書き)
    # これにより、ModelやDataModuleは自分以外のカテゴリの設定値にもアクセス可能になる
    all_params = {}
    all_params.update(user_common_params)
    all_params.update(user_data_params)
    all_params.update(user_adapter_params)
    all_params.update(user_model_params)

    # DataModuleのインスタンス化
    # 以前は combined_data_params だったが、今は all_params を渡す
    datamodule = DataModuleClass(adapter_transform=input_transform, **all_params)
    
    # 5. Modelの準備
    # Datasetのメタ情報を取得し、Adapterを通してModel用引数に変換する
    # Setupを呼んでメタ情報を確定させる
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    data_meta = {k: getattr(datamodule, k) for k in dir(datamodule) if not k.startswith('_')}
    
    # Adapter: "このデータなら、Modelにはこういう引数(in_channels=1など)を渡してね"
    model_init_args = adapter_mod.get_model_init_args(data_meta, user_adapter_params)
    
    # ユーザー設定 + 全パラメータ + 動的計算引数 をマージ
    # model_init_args(動的) が最優先されるべきだが、ユーザー設定(all_params)で上書きしたい場合もあるか？
    # 基本的には Adapter が計算した input_shape 等はユーザー設定より優先度が高い(整合性のため)とするのが自然だが、
    # 既存実装に合わせ、ユーザー設定ベース(all_params) に model_init_args を上書きする形をとる
    final_model_kwargs = {**all_params, **model_init_args}
    
    model = ModelClass(**final_model_kwargs)

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
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,          # 保存ディレクトリ (lightning_logs/checkpoints)
        every_n_epochs=5,         # 50エポックごと保存
        save_last=True,            # last.ckpt を常に保存
        monitor="val_acc",         # 監視メトリクス (モデルでlogされるval_acc)
        mode="max",                # val_acc を最大化
        save_top_k=-1              # ベスト1つだけ残す (他は自動削除)
    )

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        logger=logger,
        max_epochs=user_common_params.get("max_epochs"),
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,
        # NaNが出たら即座に停止し、発生箇所の詳細なスタックトレースを表示する
        detect_anomaly=True
    )
    
    # ckpt_path引数を渡すことで、前回の中断地点から学習を再開できる
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)

    # 完了フラグ
    with open(os.path.join(save_dir, "done"), "w") as f:
        f.write("finished")

if __name__ == "__main__":
    main()