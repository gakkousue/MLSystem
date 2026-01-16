# definitions/datasets/rcnp/plots.py
from MLsystem.utils.base_plot import BasePlot


class DatasetInfo(BasePlot):
    name = "Dataset Info"
    description = "データセットの件数やクラス分布を表示します(未実装)"

    def execute(self):
        print("Dataset info plot is not implemented yet.")
