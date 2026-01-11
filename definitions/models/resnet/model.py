# definitions/models/resnet/model.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.models import resnet18
# CONFIG_SCHEMA のインポートを削除
import os
import json

class Model(pl.LightningModule):
    # Adapter経由で整理された引数 (in_channels, num_classes) を受け取る
    def __init__(self, in_channels=3, num_classes=1000, **kwargs):
        super().__init__()
        
        # Hydra/Loaderから渡された kwargs をそのまま設定として扱う
        # デフォルト値の補完はConfigオブジェクト(Hydra)側で行われている前提
        self.conf = kwargs
        
        # ハイパーパラメータとして保存
        self.save_hyperparameters({
            **self.conf, 
            "in_channels": in_channels, 
            "num_classes": num_classes
        })

        # ResNet構築
        self.net = resnet18(weights=None) # deprecated warning回避のためNone指定
        if self.conf["pretrained"]:
            self.net = resnet18(weights="IMAGENET1K_V1")
        
        # 入力層の調整
        if in_channels != 3:
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
        
        # --- JSONへの保存処理 ---
        # trainer.default_root_dir は output/experiments/{hash} を指す
        if self.trainer and self.trainer.default_root_dir:
            save_dir = self.trainer.default_root_dir
            metrics_path = os.path.join(save_dir, "accuracy_metrics.json")
            
            # 既存データの読み込み
            data = {}
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r") as f:
                        data = json.load(f)
                except: pass
            
            # 現在のエポックの値を更新 (epochは0始まり)
            current_epoch = self.current_epoch
            data[str(current_epoch)] = avg_acc.item()
            
            # 書き込み
            with open(metrics_path, "w") as f:
                json.dump(data, f, indent=4)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.conf["lr"])