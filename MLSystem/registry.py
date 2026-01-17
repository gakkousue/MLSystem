import json
import os
import sys
import importlib.util
import inspect
from dataclasses import is_dataclass

from MLsystem.utils.config_base import BaseConfig
from MLsystem.utils.env_manager import EnvManager


class Registry:
    def __init__(self):
        # 環境変数からパスを取得
        self.registry_path = EnvManager().registry_path
        self.data = self._load()
        # プロジェクトルート解決の基準パスとして、レジストリファイルのディレクトリを使用
        self.registry_dir = os.path.dirname(self.registry_path)

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
            # プロジェクトルートが相対パスの場合、レジストリファイルの場所を基準にする
            proj_root = os.path.join(self.registry_dir, proj_root)

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

        # 継承 (parent) の処理
        parent_name = info.get("parent")
        if parent_name:
            # 親の情報を再帰的に取得
            parent_info = self.get_model_info(parent_name)
            
            # ディープコピーに近い形でマージのベースを作成
            import copy
            merged = copy.deepcopy(parent_info)

            # 親のパス系キー (suffixが _file) は、親のbase_dir基準で絶対パス化しておく
            for k, v in merged.items():
                if k.endswith("_file") and isinstance(v, str):
                    if not os.path.isabs(v):
                        merged[k] = self._resolve_path(merged.get("base_dir", ""), v)
            
            # 親の Adapters も同様に base_dir を絶対パス化しておく
            if "adapters" in merged:
                parent_base = merged.get("base_dir", "")
                # 親情報自体で base_dir が絶対パス化されていない場合（親がトップレベル定義の場合など）
                # ここで絶対パス化しておかないと、子に引き継いだ時にパスが解決できなくなる。
                # ただし parent_base が相対パスの場合、それを絶対パスにするには project_root が必要。
                # _resolve_path(parent_base, ".") で絶対パス化できる。
                
                abs_parent_base = self._resolve_path(parent_base, ".")
                
                for ad_name, ad_info in merged["adapters"].items():
                    ad_base = ad_info.get("base_dir", "")
                    if not os.path.isabs(ad_base):
                        # Modelのbase_dir（絶対パス化済み）と結合
                        # 親のAdapter定義は、親のbase_dirからの相対パス
                        ad_info["base_dir"] = os.path.join(abs_parent_base, ad_base)

            # 親情報の base_dir を絶対パス化 (子での利用に備えて)
            if "base_dir" in merged:
                merged["base_dir"] = self._resolve_path(merged["base_dir"], ".")

            # 子の情報で上書き（マージ）
            child_adapters = info.get("adapters", {})
            child_components = info.get("components", {})
            
            # ベースのadaptersに子のadaptersをマージ
            if "adapters" not in merged:
                merged["adapters"] = {}
            merged["adapters"].update(child_adapters)
            
            # componentsのマージ
            if "components" not in merged:
                merged["components"] = {}
            merged["components"].update(child_components)

            # その他のフィールドを上書き
            for k, v in info.items():
                if k not in ["adapters", "components"]:
                    merged[k] = v
            
            info = merged

        return info

    def get_dataset_info(self, dataset_name):
        info = self.data.get("datasets", {}).get(dataset_name)
        if not info:
            raise ValueError(f"Dataset '{dataset_name}' not found in registry.")
        
        # Datasetも継承できるようにする場合ここにも同様のロジックが必要だが、
        # 今回の要件は主にModelの継承/包含と思われるため、Model優先で実装。
        # 必要なら追加する。
        return info

    def get_adapter_info(self, model_name, adapter_name):
        model_info = self.get_model_info(model_name)
        adapters = model_info.get("adapters", {})
        info = adapters.get(adapter_name)
        
        if not info:
             raise ValueError(
                f"Adapter '{adapter_name}' not found in model '{model_name}'."
            )

        # Adapterのbase_dir処理
        model_base = model_info.get("base_dir", "")
        adapter_base = info.get("base_dir", "")
        
        merged_info = info.copy()
        
        # adapter_base が絶対パスならそのまま (親情報の継承で変換済み、または元から絶対パス)
        if os.path.isabs(adapter_base):
            merged_info["base_dir"] = adapter_base
        else:
            # 相対パスなら結合
            # model_base が絶対パスなら join で絶対パス、相対なら相対になる
            merged_info["base_dir"] = os.path.join(model_base, adapter_base)
        
        return merged_info

    def load_module_from_info(self, info, file_key):
        """Registry情報とキー(例: 'main_file')を指定してモジュールをロード"""
        base_dir = info.get("base_dir", "")
        filename = info.get(file_key)
        if not filename:
            raise ValueError(f"Key '{file_key}' not defined in registry info.")

        # filename が絶対パスならそれを採用、そうでなければ解決
        if os.path.isabs(filename):
            path = filename
        else:
            path = self._resolve_path(base_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Module file not found: {path}")

            # モジュール名を作成 (Pythonの標準インポート形式に合わせる)
        # 例: definitions/models/resnet/config.py -> definitions.models.resnet.config
        proj_root = os.path.normpath(self.data.get("project_root", self.registry_dir))
        if not os.path.isabs(proj_root):
            proj_root = os.path.join(self.registry_dir, proj_root)
        rel_path = os.path.relpath(path, proj_root)
        module_name = os.path.splitext(rel_path)[0].replace(os.sep, ".")

        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module  # キャッシュ登録
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
                elif (
                    base_class and issubclass(obj, base_class) and obj is not base_class
                ):
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
            if sub_name:  # Adapter
                info = self.get_adapter_info(name, sub_name)
            else:
                info = self.get_model_info(name)
        elif category == "datasets":
            info = self.get_dataset_info(name)
        else:
            raise ValueError(f"Unknown category: {category}")

        mod = self.load_module_from_info(info, "config_file")
        return self.find_class_in_module(
            mod, base_class=BaseConfig, is_dataclass_type=True
        )

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
