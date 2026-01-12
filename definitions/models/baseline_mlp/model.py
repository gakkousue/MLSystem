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
        self.lp1 = nn.Linear(7, n_units) # part dim = 7
        self.lp2 = nn.Linear(n_units, n_units)
        self.lc1 = nn.Linear(n_units + 3, n_units) # axis dim = 3
        self.lc2 = nn.Linear(n_units, n_units)
        self.lc3 = nn.Linear(n_units, n_units)
        self.lc4 = nn.Linear(n_units, n_units)
        self.lc5 = nn.Linear(n_units, n_out)
            
    def forward(self, a, p, n):
        n_bat, n_par, n_dim = p.shape 
        nonl = torch.tanh
        
        h = p.view(-1, n_dim)
        h = nonl(self.lp1(h))
        h = nonl(self.lp2(h))
        h = h.view(n_bat, n_par, -1)
        
        # Sum pooling normalized by particle number 'n'
        # n is (batch_size,), reshape to (batch_size, 1) for broadcasting
        h = torch.sum(h, dim=1) / n.view(-1, 1)

        h = torch.cat((h, a), dim=1)
        h = nonl(self.lc1(h))
        h = nonl(self.lc2(h))
        h = nonl(self.lc3(h))
        h = nonl(self.lc4(h))
        return self.lc5(h)

class Model(pl.LightningModule):
    def __init__(self, n_units=100, num_classes=3, lr=0.01, step_size=200, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.conf = kwargs
        
        self.net = BaselineModel(n_units=n_units, n_out=num_classes)
        self.lr = lr
        self.step_size = step_size
        
        # Metrics storage
        self.validation_step_outputs = []

    def forward(self, a, p, n):
        return self.net(a, p, n)

    def training_step(self, batch, batch_idx):
        # batch: (axis, part, num, label)
        a, p, n, t = batch
        outputs = self(a, p, n)
        loss = F.cross_entropy(outputs, t)
        
        # Log training loss
        self.log("train_loss", loss)
        
        # Calculate accuracy for logging
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == t).float().mean()
        self.log("train_acc", acc, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        a, p, n, t = batch
        outputs = self(a, p, n)
        loss = F.cross_entropy(outputs, t)
        
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == t).float().mean()
        
        self.validation_step_outputs.append({"val_loss": loss, "val_acc": acc})
        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if not outputs: return
            
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_acc", avg_acc, prog_bar=True)
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
                with open(metrics_path, "r") as f: data = json.load(f)
            except: pass
        data[str(self.current_epoch)] = val_acc
        with open(metrics_path, "w") as f:
            json.dump(data, f, indent=4)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=0.5)
        return [optimizer], [scheduler]