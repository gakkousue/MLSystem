# definitions/models/baseline_mlp/plots.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from MLsystem.utils.base_plot import BasePlot

class RCNPEvaluation(BasePlot):
    """
    RCNPデータセットに対する評価用プロットの基底クラス。
    推論結果の取得ロジックを共有する。
    """
    def get_predictions(self):
        # Checkpoint check
        ckpt_path = self.loader.get_checkpoint_path()
        if not ckpt_path:
            print("No checkpoint found. Triggering training...")
            self.run_training()
            ckpt_path = self.loader.get_checkpoint_path()
            if not ckpt_path: raise RuntimeError("No checkpoint after training.")

        print(f"Using checkpoint: {ckpt_path}")
        model = self.loader.load_model_from_checkpoint(ckpt_path)
        
        # Use Test Set
        _, datamodule = self.loader.setup(stage="test")
        test_loader = datamodule.test_dataloader()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        y_list = []
        t_list = []
        
        print("Running inference on test set...")
        with torch.no_grad():
            for batch in test_loader:
                # batch: a, p, n, t
                a, p, n, t = batch
                a = a.to(device)
                p = p.to(device)
                n = n.to(device)
                
                outputs = model(a, p, n)
                probs = torch.softmax(outputs, dim=1)
                
                y_list.append(probs.cpu().numpy())
                t_list.append(t.numpy())
                
        y = np.concatenate(y_list, axis=0) # (N, 3)
        t = np.concatenate(t_list, axis=0) # (N,)
        return t, y, datamodule.class_names


class ConfusionMatrixPlot(RCNPEvaluation):
    name = "Confusion Matrix"
    description = "テストデータに対する混同行列を表示します"

    def execute(self):
        t, y, class_names = self.get_predictions()
        
        preds = np.argmax(y, axis=1)
        cm = confusion_matrix(t, preds)
        
        # Plot using table (as in original code)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        
        row_labels = [f"{c} event" for c in class_names]
        col_labels = class_names
        
        table = ax.table(
            cellText=cm,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc="center",
            loc="center"
        )
        table.scale(1.2, 1.6)
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        
        plt.title("Confusion Matrix")
        
        save_path = os.path.join(self.loader.exp_dir, "confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")


class LikenessHistogram(RCNPEvaluation):
    name = "Likeness Histogram"
    description = "各クラスの確率分布ヒストグラムを表示します"
    
    def execute(self):
        t, y, class_names = self.get_predictions()
        colors = ["red", "blue", "green"]
        
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        
        for true_cls in range(3):
            mask = (t == true_cls)
            probs_true = y[mask]
            
            for pred_cls in range(3):
                ax = axes[true_cls, pred_cls]
                # bins=50, range=(0,1)
                ax.hist(probs_true[:, pred_cls], bins=50, range=(0,1), 
                        color=colors[true_cls], alpha=0.7, density=True)
                
                ax.set_xlim(0, 1)
                if pred_cls == 0:
                    ax.set_ylabel(f"{class_names[true_cls]} events")
                if true_cls == 2:
                    ax.set_xlabel(f"{class_names[pred_cls]} likeness")
                    
                ax.set_title(f"{class_names[true_cls]} -> {class_names[pred_cls]}")
                
        plt.tight_layout()
        save_path = os.path.join(self.loader.exp_dir, "likeness_histograms.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")


class ROCCurvePlot(RCNPEvaluation):
    name = "ROC Curves"
    description = "各クラスの識別性能を示すROC曲線を表示します"

    def execute(self):
        t, y, class_names = self.get_predictions()
        
        def calculate_roc_points(t, y_prob, signal_label, background_label):
            signal_probs = y_prob[t == signal_label, signal_label]
            background_probs = y_prob[t == background_label, signal_label]
            
            n_signal = len(signal_probs)
            n_background = len(background_probs)
            
            thresholds = np.linspace(0, 1, 1001)
            effs, misids = [], []
            
            for th in thresholds:
                eff = np.sum(signal_probs >= th) / n_signal if n_signal > 0 else 0.
                misid = np.sum(background_probs >= th) / n_background if n_background > 0 else 0.
                effs.append(eff)
                misids.append(misid)
                
            return np.array(effs), np.array(misids)

        for signal_idx in range(3):
            plt.figure(figsize=(7, 6))
            
            for background_idx in range(3):
                if signal_idx == background_idx: continue
                
                effs, misids = calculate_roc_points(t, y, signal_idx, background_idx)
                mask = misids > 0
                
                plt.plot(effs[mask], misids[mask], label=f'BG: {class_names[background_idx]}')
                
            plt.yscale('log')
            plt.grid(True, which="both", linestyle='--')
            plt.xlabel(f'{class_names[signal_idx]} tagging efficiency')
            plt.ylabel('Mis-identification Fraction')
            plt.title(f'ROC Curve for {class_names[signal_idx]}-tagging')
            plt.legend()
            
            save_path = os.path.join(self.loader.exp_dir, f"roc_{class_names[signal_idx]}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved: {save_path}")