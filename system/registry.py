import json
import os
import sys
import importlib.util
import inspect
from dataclasses import is_dataclass

# 環境変数を設定し、sys.pathに必要なパスを追加
import env_setup
env_setup.add_to_sys_path()

from utils.config_base import BaseConfig

class Registry:
    def __init__(self, registry_path="configs/registry.json"):
        # プロジェクトルートからの相対パスを解決するため、cwdを取得
        self.cwd = os.getcwd()
        self.registry_path = os.path.join(self.cwd, registry_path)
        self.data = self._load()

    def _load(self):
        if not os.path.exists(self.registry_path):
            raise FileNotFoundError(f"Registry file not found: {self.registry_path}")
        # UTF-8を明示して読み込む
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_path(self, base_path, rel_path):
        """パスを結合して絶対パスにする"""
        # project_root の解決
        proj_root = self.data.get("project_root", ".")
        if not os.path.isabs(proj_root):
            proj_root = os.path.join(self.cwd, proj_root)
        
        # base_dir の解決
        if not os.path.isabs(base_path):
            base_path = os.path.join(proj_root, base_path)
            
        # ターゲットファイルの解決
        full_path = os.path.join(base_path, rel_path)
        return os.path.normpath(full_path)

    def get_model_info(self, model_name):
        info = self.data.get("models", {}).get(model_name)
        if not info:
            raise ValueError(f"Model '{model_name}' not found in registry.")
        return info

    def get_dataset_info(self, dataset_name):
        info = self.data.get("datasets", {}).get(dataset_name)
        if not info:
            raise ValueError(f"Dataset '{dataset_name}' not found in registry.")
        return info
    
    def get_adapter_info(self, model_name, adapter_name):
        model_info = self.get_model_info(model_name)
        adapters = model_info.get("adapters", {})
        info = adapters.get(adapter_name)
        if not info:
            raise ValueError(f"Adapter '{adapter_name}' not found in model '{model_name}'.")
        
        # Adapterのbase_dirはModelのbase_dirからの相対パスとして扱う仕様
        # しかしJSON上ではパス文字列として結合する必要がある
        model_base = model_info.get("base_dir", "")
        adapter_base = info.get("base_dir", "")
        
        # 情報をコピーしてbase_dirを結合済みのものに書き換えて返す
        merged_info = info.copy()
        merged_info["base_dir"] = os.path.join(model_base, adapter_base)
        return merged_info

    def load_module_from_info(self, info, file_key):
        """Registry情報とキー(例: 'main_file')を指定してモジュールをロード"""
        base_dir = info.get("base_dir", "")
        filename = info.get(file_key)
        if not filename:
            raise ValueError(f"Key '{file_key}' not defined in registry info.")
            
        path = self._resolve_path(base_dir, filename)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Module file not found: {path}")
            
                # モジュール名を作成 (Pythonの標準インポート形式に合わせる)
        # 例: definitions/models/resnet/config.py -> definitions.models.resnet.config
        proj_root = os.path.normpath(self.data.get("project_root", self.cwd))
        rel_path = os.path.relpath(path, proj_root)
        module_name = os.path.splitext(rel_path)[0].replace(os.sep, ".")
        
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module # キャッシュ登録
            spec.loader.exec_module(module)
            return module
        else:
            raise ImportError(f"Could not load module from {path}")

    def find_class_in_module(self, module, base_class=None, is_dataclass_type=False):
        """モジュールから特定の条件を満たすクラスを探す"""
        candidates = []
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                # 定義元がそのモジュールであること
                # (spec_from_file_locationでロードした場合、__module__はspecで指定した名前になる)
                if obj.__module__ != module.__name__:
                    continue

                if is_dataclass_type and is_dataclass(obj):
                    candidates.append(obj)
                elif base_class and issubclass(obj, base_class) and obj is not base_class:
                    candidates.append(obj)
        
        # 優先順位などのロジックがあればここで適用
        # BaseConfig継承クラスを優先するロジック (Config用)
        if is_dataclass_type and base_class:
             for c in candidates:
                if issubclass(c, base_class):
                    return c
        
        if candidates:
            return candidates[0]
        return None

    # --- 便利メソッド群 ---

    def get_config_class(self, category, name, sub_name=None):
        if category == "models":
            if sub_name: # Adapter
                info = self.get_adapter_info(name, sub_name)
            else:
                info = self.get_model_info(name)
        elif category == "datasets":
            info = self.get_dataset_info(name)
        else:
            raise ValueError(f"Unknown category: {category}")
            
        mod = self.load_module_from_info(info, "config_file")
        return self.find_class_in_module(mod, base_class=BaseConfig, is_dataclass_type=True)

    def get_main_class(self, category, name):
        """Model(LightningModule) または DataModule クラスを取得"""
        import pytorch_lightning as pl
        
        if category == "models":
            info = self.get_model_info(name)
            mod = self.load_module_from_info(info, "main_file")
            return self.find_class_in_module(mod, base_class=pl.LightningModule)
            
        elif category == "datasets":
            info = self.get_dataset_info(name)
            mod = self.load_module_from_info(info, "main_file")
            return self.find_class_in_module(mod, base_class=pl.LightningDataModule)

    def get_adapter_module(self, model_name, adapter_name):
        info = self.get_adapter_info(model_name, adapter_name)
        return self.load_module_from_info(info, "main_file")