# definitions/models/resnet/plots.py
import os
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# 親クラスのインポート
from MLsystem.utils.base_plot import BasePlot


class ConfusionMatrix(BasePlot):
    """
    検証データに対する混同行列(Confusion Matrix)を作成するPlotジョブ。
    学習済みモデルが必要。なければ自動的に学習を行う。
    """

    name = "Confusion Matrix"
    description = "混同行列を表示します。学習済みモデルが必要です。"

    def execute(self):
        print(f"[{self.name}] Checking prerequisites...")

        # 1. チェックポイントの確認
        ckpt_path = self.loader.get_checkpoint_path()

        if not ckpt_path:
            print(f"[{self.name}] No checkpoint found. Triggering training...")
            self.run_training()

            ckpt_path = self.loader.get_checkpoint_path()
            if not ckpt_path:
                raise RuntimeError("Training finished but no checkpoint found.")

        print(f"[{self.name}] Using checkpoint: {ckpt_path}")

        # 2. モデルのロード
        model = self.loader.load_model_from_checkpoint(ckpt_path)
        model.eval()

        # 3. データセットの準備 (Validationのため fit ステージ)
        _, datamodule = self.loader.setup(stage="fit")
        val_loader = datamodule.val_dataloader()

        all_preds = []
        all_labels = []
        device = model.device

        if torch.cuda.is_available():
            model = model.to("cuda")
            device = "cuda"

        # 4. 推論実行
        print(f"[{self.name}] Running inference...")
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())

        # 5. プロット作成と保存
        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        output_dir = self.loader.exp_dir
        save_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[{self.name}] Saved: {save_path}")


class AccuracyHistory(BasePlot):
    """
    1エポックごとの検証精度(Validation Accuracy)の推移をグラフ化する。
    学習時に保存された accuracy_metrics.json を使用する。
    データが不足している場合は学習を実行してJSONを更新させる。
    """

    name = "Accuracy History"
    description = "エポックごとの精度推移を作成します。不足分は自動学習で埋めます。"

    def execute(self):
        # 目標とする最大エポック数を取得
        max_epochs = self.loader.load_modules()["model_params"].get("epochs", 5)

        # common設定からの復元を試みる (Loaderがまだcommonを返さない場合のフォールバック)
        common_diff = self.loader.diff_payload.get("common_diff", {})
        if "max_epochs" in common_diff:
            max_epochs = common_diff["max_epochs"]

        print(f"[{self.name}] Target max epochs: {max_epochs}")

        metrics_file = os.path.join(self.loader.exp_dir, "accuracy_metrics.json")
        metrics = {}

        # 1. JSONを確認
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

        # 2. データが足りているか確認 (0 ～ max_epochs-1 まで全てあるか)
        existing_epochs = [int(k) for k in metrics.keys()]
        missing_epochs = [e for e in range(max_epochs) if e not in existing_epochs]

        if missing_epochs:
            print(f"[{self.name}] Missing data for epochs: {missing_epochs}")
            print(f"[{self.name}] Triggering training to fill gaps...")

            # 不足分がある場合、目標エポックまで学習を実行
            # 学習側(Model)が on_validation_epoch_end でJSONを更新してくれる
            overrides = {"common.max_epochs": max_epochs}
            self.run_training(overrides=overrides)

            # 再読み込み
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
            else:
                raise RuntimeError("Training finished but metrics file not found.")

        # 3. グラフ描画
        # キーをintにしてソート
        epochs = sorted([int(k) for k in metrics.keys()])
        accs = [metrics[str(e)] for e in epochs]

        if not epochs:
            print(f"[{self.name}] No data to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, accs, marker="o", label="Validation Accuracy")
        plt.title(f"Accuracy History (0 to {max_epochs-1})")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        # 整数目盛りにする
        plt.xticks(range(min(epochs), max(epochs) + 1))
        plt.ylim(0, 1.0)
        plt.legend()

        save_path = os.path.join(self.loader.exp_dir, "accuracy_history.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[{self.name}] Graph saved: {save_path}")
