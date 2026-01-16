# system/inspector.py
import sys
import inspect
import importlib.util
from dataclasses import is_dataclass

# 環境変数を設定し、sys.pathに必要なパスを追加
import env_setup
env_setup.add_to_sys_path()

from utils.base_plot import BasePlot
from registry import Registry
from utils.config_base import BaseConfig

def get_available_plots(model_name, adapter_name=None, dataset_name=None):
    """
    Registry経由でPlotクラスを収集して返す。
    """
    registry = Registry()
    plot_classes = []

    def load_and_collect(info, key="plot_file"):
        try:
            mod = registry.load_module_from_info(info, key)
            
            # モジュール内のBasePlot継承クラスを収集
            for name, obj in inspect.getmembers(mod):
                if inspect.isclass(obj):
                    if issubclass(obj, BasePlot) and obj is not BasePlot:
                        if obj.__module__ == mod.__name__:
                            plot_classes.append(obj)
        except (ValueError, FileNotFoundError, ImportError):
            pass # 定義がない場合などはスキップ

    if model_name:
        try:
            info = registry.get_model_info(model_name)
            load_and_collect(info)
        except: pass
    
    if model_name and adapter_name:
        try:
            info = registry.get_adapter_info(model_name, adapter_name)
            load_and_collect(info)
        except: pass
        
    if dataset_name:
        try:
            info = registry.get_dataset_info(dataset_name)
            load_and_collect(info)
        except: pass

    return plot_classes

# find_config_class は共通設定(CommonConfig)のためにまだ必要だが、
# Registry内でも似たロジックを使うため、ここから Registry への依存は避けるか、
# あるいは Registry 側がこの関数を使わないように実装した (find_class_in_module)。
# 互換性のため残しておく。
def find_config_class(module):
    """
    モジュール内から dataclass で定義された Config クラスを探して返す。
    """

    candidates = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and is_dataclass(obj):
            if obj.__module__ == module.__name__:
                candidates.append(obj)
    
    for c in candidates:
        if issubclass(c, BaseConfig):
            return c
    
    if candidates:
        return candidates[0]
        
    return None

def find_config_class(module):
    """
    モジュール内から dataclass で定義された Config クラスを探して返す。
    system.utils.config_base.BaseConfig を継承しているものを優先するが、
    なければ単なる dataclass を探す。
    """

    candidates = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and is_dataclass(obj):
            if obj.__module__ == module.__name__:
                candidates.append(obj)
    
    # BaseConfig継承クラスを優先
    for c in candidates:
        if issubclass(c, BaseConfig):
            return c
    
    # 見つからなければ最初のdataclassを返す
    if candidates:
        return candidates[0]
        
    return None