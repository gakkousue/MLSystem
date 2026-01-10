# system/inspector.py
import importlib
import inspect
import os
import sys
from system.utils.base_plot import BasePlot

def get_available_plots(model_name):
    """
    指定されたモデルの定義ディレクトリにある plots.py を読み込み、
    BasePlotを継承したクラスの一覧を返す。
    
    return: List of class objects
    """
    # モジュールパス: definitions.models.{model_name}.plots
    module_path = f"definitions.models.{model_name}.plots"
    
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        # plots.py が無い場合は空リストを返す
        return []
        
    plot_classes = []
    
    # モジュール内の全メンバーを走査
    for name, obj in inspect.getmembers(mod):
        if inspect.isclass(obj):
            # BasePlotを継承しているか確認 (BasePlot自身は除外)
            if issubclass(obj, BasePlot) and obj is not BasePlot:
                # 定義されたモジュールが現在のモジュールと一致するものだけを取得
                # (from xxx import BasePlot などを除外するため)
                if obj.__module__ == mod.__name__:
                    plot_classes.append(obj)
                    
    return plot_classes