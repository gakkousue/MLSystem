# system/utils/base_plot.py
import subprocess
import sys
import os

class BasePlot:
    """
    すべてのPlotクラスの基底クラス。
    Loaderを通じたデータアクセスと、学習ジョブの動的実行機能を提供する。
    """
    def __init__(self, loader, job_args):
        self.loader = loader
        self.job_args = job_args # リスト形式: ["model=resnet", "dataset=mnist", ...]

    def run(self):
        """Runnerから呼ばれるエントリポイント"""
        print(f"Starting Plot Job: {self.__class__.__name__}")
        self.execute()

    def execute(self):
        """
        サブクラスで実装する描画・評価ロジック。
        この中で self.run_training() を呼び出して不足分の学習を行うことができる。
        """
        raise NotImplementedError("Subclasses must implement execute()")

    def run_training(self, overrides=None):
        """
        学習スクリプト(system/train.py)をサブプロセスで実行し、同期的に待機する。
        
        args:
            overrides: 追加・変更したいHydra引数の辞書。
                       例: {"max_epochs": 10, "model_params.save_top_k": -1}
        """
        if overrides is None:
            overrides = {}
            
        print(f">> Triggering dependent training task with overrides: {overrides}")
        
        # 既存の引数をコピー
        cmd_args = self.job_args.copy()
        
        # overridesを適用
        # 既存のキーがあれば置換、なければ追加という単純なロジックではなく、
        # Hydraの引数形式 (+key=value や key=value) に合わせて追加する。
        # 既存の引数と重複する場合、Hydraは後ろにあるものを優先するため、末尾に追加すればOK。
        
        for k, v in overrides.items():
            # Hydra形式に変換 (ネストされたキーなどはそのまま渡す前提)
            # 値がNoneの場合はキーのみ...ということはあまりないため =str(v) とする
            cmd_args.append(f"{k}={v}")
            
        # 実行コマンド構築
        cmd = [sys.executable, "system/execute_train.py"] + cmd_args
        
        # 実行
        try:
            # cwdはプロジェクトルートを想定
            ret = subprocess.run(cmd, cwd=os.getcwd())
            
            if ret.returncode != 0:
                raise RuntimeError(f"Training failed with return code {ret.returncode}")
                
            print(">> Dependent training task finished successfully.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to run training process: {e}")