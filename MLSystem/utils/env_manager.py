import os
import json
import importlib.util
import sys


class EnvManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EnvManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        config_path = os.environ.get("MLSYSTEM_CONFIG")
        if not config_path:
            raise RuntimeError("Environment variable 'MLSYSTEM_CONFIG' is not set.")

        if not os.path.exists(config_path):
            raise RuntimeError(f"Config file not found at: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to decode JSON config: {e}")

        # 必須キーの確認
        required_keys = ["common_dir", "output_dir", "queue_dir", "registry_path"]
        missing = [k for k in required_keys if k not in self.config]
        if missing:
            raise RuntimeError(f"Missing required keys in env config: {missing}")

    @property
    def common_dir(self):
        return self.config["common_dir"]

    @property
    def output_dir(self):
        return self.config["output_dir"]

    @property
    def queue_dir(self):
        return self.config["queue_dir"]

    @property
    def registry_path(self):
        return self.config["registry_path"]

    def get_common_config_module(self):
        """commonディレクトリにあるconfig.pyをモジュールとして読み込んで返す"""
        config_py_path = os.path.join(self.common_dir, "config.py")
        if not os.path.exists(config_py_path):
            raise FileNotFoundError(
                f"Common config file not found at: {config_py_path}"
            )

        module_name = "common.config"
        spec = importlib.util.spec_from_file_location(module_name, config_py_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module  # キャッシュ登録
            spec.loader.exec_module(module)
            return module
        else:
            raise ImportError(f"Could not load module from {config_py_path}")
