# system/checkpoint_manager.py
import os
import re
from typing import List, Dict, Optional, Any
from pytorch_lightning.callbacks import ModelCheckpoint

class CheckpointManager:
    """
    実験ディレクトリ内のチェックポイントの検索、一覧取得、
    およびTrainer用コールバックの作成を担当するクラス。
    """
    def __init__(self, exp_dir: str):
        self.exp_dir = exp_dir
        # チェックポイントの保存先: {exp_dir}/lightning_logs/checkpoints
        self.ckpt_dir = os.path.join(exp_dir, "lightning_logs", "checkpoints")

    def exists(self) -> bool:
        """チェックポイントディレクトリが存在するか"""
        return os.path.exists(self.ckpt_dir)

    def get_resume_path(self) -> Optional[str]:
        """
        学習再開用のチェックポイントパスを返す。
        'last.ckpt' があれば最優先、なければエポック番号が最大のものを返す。
        """
        if not self.exists():
            return None
            
        ckpts = [c for c in os.listdir(self.ckpt_dir) if c.endswith(".ckpt")]
        if not ckpts:
            return None
            
        if "last.ckpt" in ckpts:
            return os.path.join(self.ckpt_dir, "last.ckpt")
            
        # 名前でソートして一番新しいもの（簡易実装として名前順、あるいはepoch解析）
        # epoch=XX.ckpt 形式を解析してソート
        ckpts_with_epoch = []
        for c in ckpts:
            m = re.search(r"epoch=(\d+)", c)
            epoch = int(m.group(1)) if m else -1
            ckpts_with_epoch.append((epoch, c))
        
        ckpts_with_epoch.sort(key=lambda x: x[0])
        return os.path.join(self.ckpt_dir, ckpts_with_epoch[-1][1])

    def create_callback(self) -> ModelCheckpoint:
        """学習用のModelCheckpointコールバックを作成する"""
        return ModelCheckpoint(
            dirpath=self.ckpt_dir,
            every_n_epochs=5,
            save_last=True,
            monitor="val_acc",
            mode="max",
            save_top_k=-1 # 元コード準拠（-1は全保存だが、意図通りか要確認）
        )

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        利用可能なチェックポイントの一覧を取得する。
        return: List of {'epoch': int, 'path': str, 'type': 'epoch'|'last'}
        """
        if not self.exists():
            return []
            
        results = []
        # 再帰的ではなく直下のみ探索（Lightningの標準構成なら直下にある）
        for fname in os.listdir(self.ckpt_dir):
            if not fname.endswith(".ckpt"):
                continue
                
            path = os.path.join(self.ckpt_dir, fname)
            info = {"path": path, "type": "unknown", "epoch": -1}
            
            if fname == "last.ckpt":
                info["type"] = "last"
            else:
                m = re.search(r"epoch=(\d+)", fname)
                if m:
                    info["epoch"] = int(m.group(1))
                    info["type"] = "epoch"
            
            results.append(info)
        
        # エポック順にソート
        results.sort(key=lambda x: x["epoch"])
        return results

    def get_path(self, epoch: Optional[int] = None) -> Optional[str]:
        """
        指定エポックのパスを返す。Noneの場合は最新(last優先)を返す。
        """
        if epoch is None:
            return self.get_resume_path()
            
        ckpts = self.list_checkpoints()
        for c in ckpts:
            if c["epoch"] == epoch:
                return c["path"]
        return None