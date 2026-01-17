# system/inspector.py
import sys
import inspect
import importlib.util
from dataclasses import is_dataclass

from MLsystem.utils.base_plot import BasePlot
from MLsystem.registry import Registry
from MLsystem.utils.config_base import BaseConfig


def get_available_plots(model_name, adapter_name=None, dataset_name=None):
    """
    Registry経由でPlotクラスを収集して返す。
    
    Returns:
        List[Dict]: 
          [
            {"class": PlotClass, "target": None, "label": "[Model] PlotName"},
            {"class": PlotClass, "target": "backbone", "label": "[Model.backbone] PlotName"},
            ...
          ]
    """
    registry = Registry()
    plot_items = []

    def collect_from_info(info, target_member=None, label_prefix=""):
        try:
            mod = registry.load_module_from_info(info, "plot_file")
            
            # モジュール内のBasePlot継承クラスを収集
            for name, obj in inspect.getmembers(mod):
                if inspect.isclass(obj):
                    if issubclass(obj, BasePlot) and obj is not BasePlot:
                        if obj.__module__ == mod.__name__:
                            # Plotクラスのクラス変数に TITLE があればそれを名前として使う慣習があればここで使う
                            # なければクラス名
                            plot_name = getattr(obj, "TITLE", obj.__name__)
                            
                            label = f"[{label_prefix}] {plot_name}" if label_prefix else plot_name
                            
                            plot_items.append({
                                "class": obj,
                                "target": target_member,
                                "label": label
                            })
        except (ValueError, FileNotFoundError, ImportError):
            pass

    # 1. Dataset (DatasetにPlotがある場合)
    if dataset_name:
        try:
            info = registry.get_dataset_info(dataset_name)
            collect_from_info(info, label_prefix="Dataset")
        except:
            pass

    # 2. Model (Main + Parent + Components)
    if model_name:
        try:
            # Model情報の取得 (継承/包含処理済み)
            model_info = registry.get_model_info(model_name)
            
            # (A) 自分自身 (継承元のPlotファイルも model_info["plot_file"] に入っていればロードされる)
            # 継承ロジックにより、親のplot_fileがコピーされていればここで読まれる。
            # しかし、親が plot_file を持ち、子も plot_file を持つ場合、子は上書きしている。
            # 両方のPlotを使いたい場合、継承ロジックで plot_files リストにするなどの対応が必要だが、
            # 現状のRegistry仕様は単一ファイル。
            # 親のPlotも使いたいなら、親コンポーネントとして扱うか、
            # 別途「親の定義」をロードする必要がある。
            # 今回の要件「継承(parent)を辿り、親モデルのPlotも収集」に従い、明示的に親を辿る。
            
            # 自分自身のPlot
            collect_from_info(model_info, target_member=None, label_prefix="Model")
            
            # (B) 親モデルのPlot (parentを辿る)
            current_info = model_info
            current_model_name = model_name
            while "parent" in current_info:
                parent_name = current_info["parent"]
                try:
                    # 親の情報を取得（再帰マージなしの生データが欲しいが、get_model_infoはマージ済みを返す）
                    # マージ済みでも plot_file が親のものになっていればOKだが、子が上書きしていると消える。
                    # Registry.dataを直接見るのはカプセル化違反だが、親の生の定義を知る必要がある。
                    parent_raw_info = registry.data.get("models", {}).get(parent_name)
                    if parent_raw_info:
                        # 親の plot_file を使って収集
                        # ただしパス解決のために get_model_info を使う方が安全（絶対パス化など）
                        # 妥協案: 親モデル名で get_model_info して、その plot_file をロード。
                        # target_member=None (Mainモデルに対して実行)
                        parent_full_info = registry.get_model_info(parent_name)
                        collect_from_info(parent_full_info, target_member=None, label_prefix=f"Parent({parent_name})")
                        
                        current_info = parent_raw_info # 次の親へ
                    else:
                        break
                except:
                    break

            # (C) 包含 (components)
            components = model_info.get("components", {})
            for member_name, comp_registry_name in components.items():
                try:
                    comp_info = registry.get_model_info(comp_registry_name)
                    collect_from_info(comp_info, target_member=member_name, label_prefix=f"Comp.{member_name}")
                except:
                    pass

        except:
            pass

    # 3. Adapter
    if model_name and adapter_name:
        try:
            info = registry.get_adapter_info(model_name, adapter_name)
            collect_from_info(info, label_prefix="Adapter")
        except:
            pass

    return plot_items


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
