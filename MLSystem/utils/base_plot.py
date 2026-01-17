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
        self.job_args = job_args  # リスト形式: ["model=resnet", "dataset=mnist", ...]
        self.target_model = None

    @property
    def model(self):
        """
        Plot対象のモデルを返す。
        target_modelが設定されていればそれ（サブモジュール等）、
        なければLoaderが持つMainモデルを返す。
        """
        if self.target_model is not None:
            return self.target_model
        return self.loader.model

    def run(self):
        """Runnerから呼ばれるエントリポイント"""
        print(f"Starting Plot Job: {self.__class__.__name__}")
        self.execute()

    def execute(self):
        """
        サブクラスで実装する描画・評価ロジック。
        この中で self.model を使用して推論等を行う。
        """
        raise NotImplementedError("Subclasses must implement execute()")

    def run_training(self, overrides=None):
        """
        【廃止】Plotモードからの学習トリガーは廃止されました。
        学習が完了していないチェックポイントを使用しようとした場合などは、
        事前に学習ジョブを実行してください。
        """
        raise RuntimeError(
            "Automatic training trigger is deprecated. "
            "Please run a training job first if checkpoints are missing."
        )
