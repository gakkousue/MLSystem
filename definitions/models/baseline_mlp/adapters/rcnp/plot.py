# definitions/models/baseline_mlp/adapters/rcnp/plot.py
from system.utils.base_plot import BasePlot

class NoPlot(BasePlot):
    name = "No Plot Available"
    def execute(self):
        print("No adapter specific plots.")