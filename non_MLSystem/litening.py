import sys
import os
import pytorch_lightning as pl

# -----------------------------------------------------------------------------
# [ユーザー設定エリア] ここだけ書き換えてください
# -----------------------------------------------------------------------------

# 1. 使用するクラスのインポート
#    (例: ResNet + MNIST の場合)
from definitions.models.resnet.model import Model as TargetModel
from definitions.datasets.mnist.datamodule import DataModule as TargetDataModule
from definitions.models.resnet.adapters.mnist.adapter import (
    get_input_transform,
    get_model_init_args
)

# 2. パラメータ設定 (config_diff.json と user_config.json の内容をマージ)
PARAMS = {
    # Common
    "batch_size": 32,
    "seed": 42,
    "num_workers": 2,
    "gpus": 1,
    
    # Dataset / Adapter / Model Specifics
    "data_dir": "./data",
    "val_ratio": 0.2,
    "resize_scale": 1.0,
    "num_layers": 18,
    "pretrained": False,
    "lr": 0.001,
}

# 3. チェックポイントのパス (Noneの場合はランダム初期化)
CKPT_PATH = "output/experiments/xxxxxxxxxx/lightning_logs/checkpoints/epoch=9.ckpt"

# -----------------------------------------------------------------------------
# [メイン処理] 以下は変更不要です
# -----------------------------------------------------------------------------

def main():
    sys.path.append(os.getcwd())
    pl.seed_everything(PARAMS.get("seed", 42))

    print(">> [Lightning Pattern] Setting up experiment...")

    # 1. Adapter & DataModule Setup
    print(">> Initializing DataModule...")
    input_transform = get_input_transform(PARAMS)
    datamodule = TargetDataModule(adapter_transform=input_transform, **PARAMS)

    # 2. Extract Metadata from DataModule
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    
    data_meta = {k: getattr(datamodule, k) for k in dir(datamodule) if not k.startswith("_")}

    # 3. Model Initialization Args
    model_init_args = get_model_init_args(data_meta, PARAMS)
    final_model_kwargs = {**PARAMS, **model_init_args}

    # 4. Load Model
    if CKPT_PATH and os.path.exists(CKPT_PATH):
        print(f">> Loading from checkpoint: {CKPT_PATH}")
        model = TargetModel.load_from_checkpoint(checkpoint_path=CKPT_PATH, **final_model_kwargs)
    else:
        print(">> Checkpoint not found or not specified. Using random initialization.")
        model = TargetModel(**final_model_kwargs)

    # 5. Execution (Test/Inference)
    print(">> Starting Execution...")
    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
    
    # テストセットでの評価を実行
    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    main()