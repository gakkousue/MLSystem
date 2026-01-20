# MLSystem/execute_train.py
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import PyPathManager # パス解決 (import時に実行される)

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # 1. データの準備 (Hydra Instantiate)
    print(">> Instantiating DataModule...")
    datamodule = hydra.utils.instantiate(cfg.dataset)
    datamodule.prepare_data()
    datamodule.setup("fit")

    # 2. モデルの準備 (Hydra Instantiate)
    print(">> Instantiating Model...")
    model = hydra.utils.instantiate(cfg.model)

    # 3. コールバックの準備
    callbacks = []
    
    # チェックポイント保存
    # Hydraの出力ディレクトリは hydra.run.dir で指定された場所
    # デフォルトでは .hydra/ もそこに作られる
    ckpt_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="{epoch}-{val_loss:.2f}",
        save_last=True
    )
    callbacks.append(ckpt_callback)

    # 4. Trainerの準備
    # Trainerの設定もHydra化できるが、まずはシンプルにコードで記述
    # 必要に応じて cfg.trainer を参照するように拡張可能
    trainer = pl.Trainer(
        max_epochs=cfg.get("max_epochs", 10),
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        default_root_dir=os.getcwd(), # Hydraがカレントディレクトリを出力先に変更しているため cwd でOK
        enable_progress_bar=True,
    )

    
    # 6. 学習実行
    ckpt_path = None
    if cfg.get("source_run_dir"):
        # 過去の実験から再開する場合
        # source_run_dir/checkpoints/last.ckpt を探す
        potential_ckpt = os.path.join(cfg.source_run_dir, "lightning_logs", "version_0", "checkpoints", "last.ckpt")
        # または直下の checkpoints/last.ckpt (構成による)
        if not os.path.exists(potential_ckpt):
             potential_ckpt = os.path.join(cfg.source_run_dir, "checkpoints", "last.ckpt")
        
        if os.path.exists(potential_ckpt):
            ckpt_path = potential_ckpt
            print(f">> Resuming from checkpoint: {ckpt_path}")
        else:
            print(f"[Warning] Source run dir specified but checkpoint not found: {cfg.source_run_dir}")

    trainer.fit(model, datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
