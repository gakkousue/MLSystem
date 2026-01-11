# system/inspector.py
import importlib
import inspect
import os
import sys
from system.utils.base_plot import BasePlot

def get_available_plots(model_name, adapter_name=None, dataset_name=None):
    """
    Model, Adapter, Dataset の定義ディレクトリにある Plotクラスを収集して返す。
    return: List of class objects
    """
    targets = []
    
    # 1. Model Plots: definitions.models.{model}.plots
    if model_name:
        targets.append(f"definitions.models.{model_name}.plots")
        
    # 2. Adapter Plots: definitions.models.{model}.adapters.{adapter}.plot
    if model_name and adapter_name:
        targets.append(f"definitions.models.{model_name}.adapters.{adapter_name}.plot")
        
    # 3. Dataset Plots: definitions.datasets.{dataset}.plots
    if dataset_name:
        targets.append(f"definitions.datasets.{dataset_name}.plots")

    plot_classes = []

    for module_path in targets:
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            continue
            
        # モジュール内の全メンバーを走査
        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj):
                # BasePlotを継承しているか確認 (BasePlot自身は除外)
                if issubclass(obj, BasePlot) and obj is not BasePlot:
                    # 定義されたモジュールが現在のモジュールと一致するものだけを取得
                    if obj.__module__ == mod.__name__:
                        plot_classes.append(obj)
                        
    return plot_classes