# system/utils/hydra_helper.py
import sys
import os
from typing import Dict, List, Any
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

def dict_to_hydra_args(config: Dict[str, Any]) -> List[str]:
    """
    config.json (辞書) から Hydraのコマンドライン引数リストを生成する。
    """
    args = []

    # 1. 基本構成 (defaults)
    for key in ["model", "adapter", "dataset"]:
        if key in config and isinstance(config[key], str):
            args.append(f"{key}={config[key]}")

    # 2. パラメータグループ
    target_groups = ["common", "model_params", "adapter_params", "data_params"]

    for group in target_groups:
        if group in config and isinstance(config[group], dict):
            for k, v in config[group].items():
                if k.startswith("_"):
                    continue
                
                if isinstance(v, bool):
                    val_str = str(v).lower()
                elif v is None:
                    val_str = "null"
                else:
                    val_str = str(v)
                
                # 上書き形式で追加
                args.append(f"+{group}.{k}={val_str}")

    return args


def get_config_from_args(args: List[str], config_path: str = "../configs", config_name: str = "config"):
    """
    Hydra引数リストから DictConfig オブジェクトを生成する。
    """
    # GlobalHydraが初期化されていなければ初期化する
    if not GlobalHydra.instance().is_initialized():
        # hydra.initialize は呼び出し元スクリプトからの相対パスを期待する
        # config_path はこのファイル(hydra_helper.py)からの相対パスで、
        # MLSystemパッケージ内のconfigsディレクトリを指定する
        # __file__ はこのファイルのパス, ".." でutilsの一つ上(MLSystem)へ, "configs" で目的のディレクトリへ
        rel_config_path = os.path.join(os.path.dirname(__file__), "..", "configs")
        
        from hydra import initialize_config_dir
        # initialize_config_dirには絶対パスを指定するのが最も安全
        abs_config_path = os.path.abspath(rel_config_path)
        initialize_config_dir(config_dir=abs_config_path, version_base=None)

    # overridesの作成
    return compose(config_name=config_name, overrides=args)