# definitions/models/resnet/model.py
import os
import json
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18

class Model(pl.LightningModule):
    def __init__(self, in_channels=3, num_classes=1000, lr=0.001, pretrained=False):
        """
        ResNet Model
        Args:
            in_channels (int): 入力チャンネル数
            num_classes (int): クラス数
            lr (float): 学習率
            pretrained (bool): ImageNet事前学習済みの重みを使用するか
        """
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        
        # ResNet構築
        # weights引数の指定方法はtorchvisionのバージョンによるが、最近はweights="IMAGENET1K_V1"推奨
        weights = "IMAGENET1K_V1" if pretrained else None
        self.net = resnet18(weights=weights)

        # 入力層の調整
        if in_channels != 3:
            # ResNetの最初のConv2dは (3, 64, kernel=7, stride=2, padding=3, bias=False)
            self.net.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # 出力層の調整
        if self.net.fc.out_features != num_classes:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

        # 検証用の一時保存リスト
        self.validation_step_outputs = []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """1バッチごとの検証処理"""
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # 集計用に値を保存
        self.validation_step_outputs.append({"val_loss": loss, "val_acc": acc})
        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self):
        """エポック終了時の集計と保存"""
        outputs = self.validation_step_outputs
        if not outputs:
            return

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        # Lightningのログに記録
        self.log("val_loss", avg_loss)
        self.log("val_acc", avg_acc)

        self.validation_step_outputs.clear()

        # --- JSONへの保存処理 (簡易的な記録) ---
        # trainer.default_root_dir は Hydraの出力ディレクトリ等を指す
        if self.trainer and self.trainer.default_root_dir:
            save_dir = self.trainer.default_root_dir
            metrics_path = os.path.join(save_dir, "accuracy_metrics.json")
            
            # 簡易排他制御なしで読み書き（並列実行時は注意が必要だが、学習プロセスは1つ前提）
            data = {}
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r") as f:
                        data = json.load(f)
                except:
                    pass

            current_epoch = self.current_epoch
            data[str(current_epoch)] = avg_acc.item()

            with open(metrics_path, "w") as f:
                json.dump(data, f, indent=4)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
