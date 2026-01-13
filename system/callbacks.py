# system/callbacks.py
import pytorch_lightning as pl
import torch
from rich.console import Console
from rich.table import Table

console = Console()

class JobLoggingCallback(pl.Callback):
    """
    学習進捗を JobLog 形式で出力。
    enable_progress_bar=False の環境でも self.log で保存した全てのメトリクスを
    rich でテーブル表示して見やすくする。
    """
    
    def on_train_start(self, trainer, pl_module):
        print(f"[JobLog] Training started. Max epochs: {trainer.max_epochs}")

    def on_train_end(self, trainer, pl_module):
        print(f"[JobLog] Training finished.")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Sanity Check中はスキップ
        if trainer.sanitizing:
            return

        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics

        # rich Table作成
        table = Table(title=f"Epoch {epoch} Metrics")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        # metricsをソートして出力
        for k in sorted(metrics.keys()):
            v = metrics[k]
            if isinstance(v, torch.Tensor):
                v = v.item()
            try:
                v = float(v)
                v_str = f"{v:.4f}"
            except:
                v_str = str(v)
            table.add_row(k, v_str)

        console.print(table)
