import sys
import os
import torch

# -----------------------------------------------------------------------------
# [ユーザー設定エリア] ここだけ書き換えてください
# -----------------------------------------------------------------------------

# 1. 使用するクラスのインポート
from definitions.models.resnet.model import Model as TargetModel
from definitions.datasets.mnist.datamodule import DataModule as TargetDataModule
from definitions.models.resnet.adapters.mnist.adapter import (
    get_input_transform,
    get_model_init_args
)

# 2. パラメータ設定
PARAMS = {
    # Common
    "batch_size": 32,
    "seed": 42,
    "num_workers": 0, # Windows等でエラーが出る場合は0推奨
    
    # Dataset / Adapter / Model Specifics
    "data_dir": "./data",
    "val_ratio": 0.2,
    "resize_scale": 1.0,
    "num_layers": 18,
    "pretrained": False,
    "lr": 0.001,
}

# 3. チェックポイントのパス
CKPT_PATH = "output/experiments/xxxxxxxxxx/lightning_logs/checkpoints/epoch=9.ckpt"

# -----------------------------------------------------------------------------
# [メイン処理] 以下は変更不要です
# -----------------------------------------------------------------------------

def main():
    sys.path.append(os.getcwd())
    torch.manual_seed(PARAMS.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f">> [Pure PyTorch Pattern] Running on {device}...")

    # 1. Adapter & DataModule Setup
    print(">> Initializing DataModule...")
    input_transform = get_input_transform(PARAMS)
    datamodule = TargetDataModule(adapter_transform=input_transform, **PARAMS)

    # 2. Prepare Data Loader
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    # ここではテスト用ローダーを取得（学習時は train_dataloader() を使用）
    data_loader = datamodule.test_dataloader()

    # 3. Model Setup
    data_meta = {k: getattr(datamodule, k) for k in dir(datamodule) if not k.startswith("_")}
    model_init_args = get_model_init_args(data_meta, PARAMS)
    final_model_kwargs = {**PARAMS, **model_init_args}

    print(">> Initializing Model...")
    model = TargetModel(**final_model_kwargs)

    # 4. Load Weights Manually
    if CKPT_PATH and os.path.exists(CKPT_PATH):
        print(f">> Loading weights from: {CKPT_PATH}")
        checkpoint = torch.load(CKPT_PATH, map_location=device)
        # Lightningのckptは "state_dict" キーの中に重みがある
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)
    else:
        print(">> Warning: Checkpoint not found. Running with random weights.")

    model.to(device)
    model.eval()

    # 5. Manual Inference Loop
    print(">> Starting Inference Loop...")
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # LightningModuleのDatasetは通常 (x, y) または (x, ...) を返す
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1] # 必要に応じてインデックス調整
            else:
                x = batch
                y = None

            x = x.to(device)
            if y is not None:
                y = y.to(device)

            # Forward
            outputs = model(x)
            
            # --- ここから下はタスク依存の評価ロジック (例: 分類タスク) ---
            if y is not None:
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            if batch_idx == 0:
                print(f"   Batch 0 Sample Output: {outputs[0][:5]}")

    if total > 0:
        print(f">> Accuracy: {100 * correct / total:.2f}%")
    else:
        print(">> Inference finished (No labels for accuracy calculation).")

if __name__ == "__main__":
    main()