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
        # 相対パス解決のため、呼び出し元の位置を考慮
        # ここではMLSystemパッケージ内から呼ばれる前提で、configsへのパスを指定
        # args[0]などのスクリプト位置に依存しないよう注意が必要
        # env_config等を使うのがベストだが、ここでは引数のデフォルト値または相対パスを使用
        
        # hydra.initialize は呼び出し元スクリプトからの相対パスを期待する
        # ここでは絶対パス変換して指定するアプローチをとる
        # config_path が "../configs" の場合、このファイル(hydra_helper.py)からの相対として解決を試みる
        
        abs_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), config_path))
        
        # RelPath計算 (hydra.initializeのversion_base=None指定で絶対パスも通る場合があるが、
        # 公式推奨は相対パス。しかしディレクトリ構造が変わると壊れる。
        # ここでは initialize_config_dir を使用して絶対パスを指定する)
        from hydra import initialize_config_dir
        initialize_config_dir(config_dir=abs_config_path, version_base=None)

    # overridesの作成
    return compose(config_name=config_name, overrides=args)