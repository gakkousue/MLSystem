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

from MLsystem.hashing import compute_combined_hash
from MLsystem.utils.env_manager import EnvManager
from MLsystem.registry import Registry
from MLsystem.inspector import find_config_class  # common設定用に残す
from MLsystem.builder import ExperimentBuilder
from MLsystem.checkpoint_manager import CheckpointManager
from MLsystem.callbacks import JobLoggingCallback  # 追加


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config):
    print(
        f"Loaded Config: Model={config.model}, Adapter={config.adapter}, Dataset={config.dataset}"
    )

    # 1. ExperimentBuilderを使用して環境を構築
    #    (クラスロード, パラメータマージ, ハッシュ計算, インスタンス化)
    try:
        builder = ExperimentBuilder(config)
        context = builder.build()
    except Exception as e:
        print(f"Error building experiment: {e}")
        return

    save_dir = os.path.join(EnvManager().output_dir, "experiments", context.hash_id)
    print(f"Experiment Hash ID: {context.hash_id}")

    os.makedirs(save_dir, exist_ok=True)

    # 2. 設定の保存
    # (A) 差分設定
    with open(os.path.join(save_dir, "config_diff.json"), "w") as f:
        json.dump(context.diff_payload, f, indent=4)

    # (B) 完全設定 (config.json)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(context.full_config, f, indent=4)

    # 3. ロガーとチェックポイント管理
    logger = pl.loggers.TensorBoardLogger(
        save_dir=save_dir, name="lightning_logs", version=""
    )

    checkpoint_manager = CheckpointManager(save_dir)

    # 再開用チェックポイントの取得
    checkpoint_path = checkpoint_manager.get_resume_path()
    if checkpoint_path:
        print(f">> Found checkpoint. Resuming from: {checkpoint_path}")

        # すでに学習完了済みかチェックして、完了していれば正常終了する
        max_epochs = context.all_params.get("max_epochs")
        if max_epochs is not None:
            try:
                # チェックポイントをCPUにロードしてエポック数を確認
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                saved_epoch = checkpoint.get("epoch", -1)

                # saved_epochは0始まりの完了エポックインデックス (例: epoch=9 は10エポック目完了)
                # したがって、学習済みエポック数は saved_epoch + 1
                finished_epochs = saved_epoch + 1

                if finished_epochs >= max_epochs:
                    print(
                        f"[DONE] [Skip] Training already reached max_epochs ({max_epochs}). Exiting."
                    )
                    sys.exit(0)
            except Exception as e:
                print(f"[Warning] Failed to inspect checkpoint: {e}")

    # 4. 学習実行
    trainer = pl.Trainer(
        default_root_dir=save_dir,
        logger=logger,
        max_epochs=context.all_params.get("max_epochs"),
        accelerator="auto",
        devices=1,
        callbacks=[
            checkpoint_manager.create_callback(),
            JobLoggingCallback(),
        ],  # Managerとログ用Callbackを設定
        enable_progress_bar=False,
        detect_anomaly=True,
    )

    # max_epochsに達している場合、Lightningは自動的に学習をスキップして終了する
    trainer.fit(context.model, context.datamodule, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    main()
