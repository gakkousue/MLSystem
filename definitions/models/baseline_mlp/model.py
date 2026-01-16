# definitions/models/baseline_mlp/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os
import json


class BaselineModel(nn.Module):
    """元のBaselineModelの実装"""

    def __init__(self, n_units=100, n_out=3):
        super(BaselineModel, self).__init__()
        self.lp1 = nn.Linear(7, n_units)  # part dim = 7
        self.lp2 = nn.Linear(n_units, n_units)
        self.lc1 = nn.Linear(n_units + 3, n_units)  # axis dim = 3
        self.lc2 = nn.Linear(n_units, n_units)
        self.lc3 = nn.Linear(n_units, n_units)
        self.lc4 = nn.Linear(n_units, n_units)
        self.lc5 = nn.Linear(n_units, n_out)

    def forward(self, a, p, n):
        # 異常値を検知してエラーを上げるヘルパー関数
        def check(tensor, location):
            if torch.isnan(tensor).any():
                raise RuntimeError(f"⚠️ NaN detected at: {location}")
            if torch.isinf(tensor).any():
                raise RuntimeError(f"⚠️ Inf detected at: {location}")

        n_bat, n_par, n_dim = p.shape
        nonl = torch.relu  # 以前はtanhでしたがreluになっていますね（そのままでOKです）

        # 入力値のチェック
        check(p, "input p")
        check(a, "input a")

        h = p.view(-1, n_dim)
        h = nonl(self.lp1(h))
        h = nonl(self.lp2(h))
        h = h.view(n_bat, n_par, -1)

        check(h, "after lp layers")

        # Sum pooling normalized by particle number 'n'
        # n is (batch_size,), reshape to (batch_size, 1) for broadcasting

        # # ここで粒子数が0のものがないかチェックし、あれば警告を出す
        # if (n == 0).any():
        #     print(f"[Warning] n=0 detected in batch! Indices: {torch.where(n==0)[0]}")

        # ゼロ除算防止のために微小値を足して割る
        numerator = torch.sum(h, dim=1)
        denominator = n.view(-1, 1) + 1e-6

        h = numerator / denominator

        # 割り算直後のチェック（ここが一番危険）
        check(h, "after pooling (division)")

        h = torch.cat((h, a), dim=1)
        h = nonl(self.lc1(h))
        check(h, "after lc1")

        h = nonl(self.lc2(h))
        h = nonl(self.lc3(h))
        h = nonl(self.lc4(h))

        out = self.lc5(h)
        check(out, "final output")

        return out


class Model(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        # パラメータの抽出と設定
        self.setup_params(**kwargs)

        # BaselineModelの初期化には抽出した変数を使用
        self.net = BaselineModel(n_units=self.n_units, n_out=self.num_classes)

        # Metrics storage
        self.validation_step_outputs = []

    def setup_params(
        self, n_units=100, num_classes=3, lr=0.01, step_size=200, **kwargs
    ):
        """
        モデルに必要なパラメータのみを定義し、デフォルト値を設定する。
        不要なkwargsは無視される。
        """
        self.n_units = n_units
        self.num_classes = num_classes
        self.lr = lr
        self.step_size = step_size

        # 必要な変数のみをhparams.yamlに保存
        self.save_hyperparameters(
            {
                "n_units": n_units,
                "num_classes": num_classes,
                "lr": lr,
                "step_size": step_size,
            }
        )

    def forward(self, a, p, n):
        return self.net(a, p, n)

    def training_step(self, batch, batch_idx):
        # batch: (axis, part, num, label)
        a, p, n, t = batch
        outputs = self(a, p, n)
        loss = F.cross_entropy(outputs, t)

        # Calculate accuracy for logging
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == t).float().mean()

        # [DEBUG] 学習状況の詳細出力
        # 現在のエポック、バッチ番号、ロス、精度、バッチサイズを表示
        # print(f"[Train] Epoch={self.current_epoch} Batch={batch_idx} Size={len(t)} | Loss={loss.item():.4f} Acc={acc.item():.4f}")

        # 手動NaNチェック: Lossが数値でない場合は即座に例外を投げる
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"エラー: Epoch={self.current_epoch}, Batch={batch_idx} でLossがNaNになりました。入力データが正規化されていない可能性があります。"
            )

        # Log training loss
        # stepごとではなくエポック終了時にまとめて平均値を記録・保存する
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )
        self.log(
            "train_acc", acc, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        a, p, n, t = batch
        outputs = self(a, p, n)
        loss = F.cross_entropy(outputs, t)

        preds = torch.argmax(outputs, dim=1)
        acc = (preds == t).float().mean()

        # # [DEBUG] 検証状況の詳細出力
        # print(f"[Train] Epoch={self.current_epoch} Batch={batch} Batchid={batch_idx} Size={len(t)} | Loss={loss.item():.4f} Acc={acc.item():.4f}")

        self.validation_step_outputs.append({"val_loss": loss, "val_acc": acc})
        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if not outputs:
            print(f"[Val End] Epoch={self.current_epoch}: No outputs recorded.")
            return

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        # [DEBUG] 検証全体の集計結果出力
        # 合計何バッチ処理したか(Steps)が重要
        print(
            f"[Val End] Epoch={self.current_epoch}: Steps={len(outputs)} AvgLoss={avg_loss.item():.4f} AvgAcc={avg_acc.item():.4f}"
        )

        # ロガーへも確実に保存する
        self.log("val_loss", avg_loss, prog_bar=True, logger=True)
        self.log("val_acc", avg_acc, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()

        # Save metrics to JSON for plotting
        if self.trainer and self.trainer.default_root_dir:
            self._save_metrics(avg_acc.item())

    def _save_metrics(self, val_acc):
        save_dir = self.trainer.default_root_dir
        metrics_path = os.path.join(save_dir, "accuracy_metrics.json")
        data = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    data = json.load(f)
            except:
                pass
        data[str(self.current_epoch)] = val_acc
        with open(metrics_path, "w") as f:
            json.dump(data, f, indent=4)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=0.5
        )
        return [optimizer], [scheduler]
