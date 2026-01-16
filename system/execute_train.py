# system/execute_train.py
import os
import sys
import json
import importlib
import torch  # 追加
import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# 環境変数を設定し、sys.pathに必要なパスを追加
import env_setup
env_setup.add_to_sys_path()

from hashing import compute_combined_hash
from registry import Registry
from inspector import find_config_class  # common設定用に残す
from builder import ExperimentBuilder
from checkpoint_manager import CheckpointManager
from callbacks import JobLoggingCallback  # 追加

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    print(f"Loaded Config: Model={cfg.model}, Adapter={cfg.adapter}, Dataset={cfg.dataset}")

    # 1. ExperimentBuilderを使用して環境を構築
    #    (クラスロード, パラメータマージ, ハッシュ計算, インスタンス化)
    try:
        builder = ExperimentBuilder(cfg)
        ctx = builder.build()
    except Exception as e:
        print(f"Error building experiment: {e}")
        return

    save_dir = os.path.join("output", "experiments", ctx.hash_id)
    print(f"Experiment Hash ID: {ctx.hash_id}")

    # 実行済みチェックを行う場合はここで
    # if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "done")): ...
    
    os.makedirs(save_dir, exist_ok=True)

    # 2. 設定の保存
    with open(os.path.join(save_dir, "config_diff.json"), "w") as f:
        json.dump(ctx.diff_payload, f, indent=4)

    # 3. ロガーとチェックポイント管理
    logger = pl.loggers.TensorBoardLogger(
        save_dir=save_dir,
        name="lightning_logs",
        version="" 
    )


    ckpt_manager = CheckpointManager(save_dir)
    
    # 再開用チェックポイントの取得
    ckpt_path = ckpt_manager.get_resume_path()
    if ckpt_path:
        print(f">> Found checkpoint. Resuming from: {ckpt_path}")

        # すでに学習完了済みかチェックして、完了していれば正常終了する
        max_epochs = ctx.all_params.get("max_epochs")
        if max_epochs is not None:
            try:
                # チェックポイントをCPUにロードしてエポック数を確認
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                saved_epoch = checkpoint.get("epoch", -1)
                
                # saved_epochは0始まりの完了エポックインデックス (例: epoch=9 は10エポック目完了)
                # したがって、学習済みエポック数は saved_epoch + 1
                finished_epochs = saved_epoch + 1
                
                if finished_epochs >= max_epochs:
                    print(f"[DONE] [Skip] Training already reached max_epochs ({max_epochs}). Exiting.")
                    sys.exit(0)
            except Exception as e:
                print(f"[Warning] Failed to inspect checkpoint: {e}")

    # 4. 学習実行
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        logger=logger,
        max_epochs=ctx.all_params.get("max_epochs"),
        accelerator="auto",
        devices=1,
        callbacks=[
            ckpt_manager.create_callback(),
            JobLoggingCallback()
        ], # Managerとログ用Callbackを設定
        enable_progress_bar=False,
        detect_anomaly=True
    )
    
    # max_epochsに達している場合、Lightningは自動的に学習をスキップして終了する
    trainer.fit(ctx.model, ctx.datamodule, ckpt_path=ckpt_path)

    # doneファイルの作成は廃止
    # 正常終了すればExit Code 0が返り、Runnerがそれを検知してジョブ完了とする

if __name__ == "__main__":
    main()