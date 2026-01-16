import os
import sys
import json
import tempfile
import shutil
import subprocess

def create_dummy_environment(base_dir):
    """ダミーの環境（ディレクトリ構成と設定ファイル）を作成する"""
    dirs = {
        "common": os.path.join(base_dir, "common"),
        "output": os.path.join(base_dir, "output"),
        "queue": os.path.join(base_dir, "queue"),
        "configs": os.path.join(base_dir, "configs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # env_config.json
    env_config = {
        "common_dir": dirs["common"],
        "output_dir": dirs["output"],
        "queue_dir": dirs["queue"],
        "registry_path": os.path.join(dirs["configs"], "registry.json")
    }
    env_config_path = os.path.join(base_dir, "env_config.json")
    with open(env_config_path, "w") as f:
        json.dump(env_config, f)

    # registry.json
    registry_data = {
        "project_root": "..", # base_dir relative to configs
        "models": {},
        "datasets": {}
    }
    with open(env_config["registry_path"], "w") as f:
        json.dump(registry_data, f)

    # common/config.py
    common_config_code = """
from dataclasses import dataclass
@dataclass
class CommonConfig:
    max_epochs: int = 10
    _name: str = "common"
"""
    with open(os.path.join(dirs["common"], "config.py"), "w") as f:
        f.write(common_config_code)

    return env_config_path

def verify_system():
    # テスト用のディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Creating dummy environment in {temp_dir}")
        env_config_path = create_dummy_environment(temp_dir)
        
        # 環境変数を設定
        env = os.environ.copy()
        env["MLSYSTEM_CONFIG"] = env_config_path
        
        # Pythonスクリプトをサブプロセスで実行し、インポートと初期化をテスト
        # カレントディレクトリは temp_dir に移動し、CWD依存がないか確認
        script_code = """
import sys
import os
try:
    from MLsystem.utils.env_manager import EnvManager
    print("EnvManager imported.")
    
    mgr = EnvManager()
    print(f"Common Dir: {mgr.common_dir}")
    print(f"Registry Path: {mgr.registry_path}")
    
    # Common Config Module loading
    mod = mgr.get_common_config_module()
    print(f"Loaded common module: {mod}")
    if not hasattr(mod, 'CommonConfig'):
        raise ImportError("CommonConfig class not found in loaded module")

    # Registry loading
    from MLsystem.registry import Registry
    reg = Registry()
    print(f"Registry data: {reg.data}")
    
    # Builder import (checks dependency chain)
    # Mocking Registry to avoid loading actual models which don't exist in dummy
    # But just importing builder shouldn't fail if we don't instantiate it
    import MLsystem.builder
    print("Builder imported.")

    print("VERIFICATION SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        script_path = os.path.join(temp_dir, "test_script.py")
        with open(script_path, "w") as f:
            f.write(script_code)

        print("Running verification script...")
        # MLSystemパッケージが見えるように、現在のsys.path[0] (c:\Users\soush\研究室\MLSystem) をPYTHONPATHに追加
        # ただし、今回はこのスクリプト自体がルートで実行されているので、
        # sys.version_info等から推測するか、単純に os.getcwd() (project root) を使う
        cwd = os.getcwd()
        env["PYTHONPATH"] = cwd + os.pathsep + env.get("PYTHONPATH", "")

        # CWDをtemp_dirに変えて実行（CWD非依存の証明）
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=temp_dir,
            env=env,
            capture_output=True,
            text=True
        )
        
        print("--- Output ---")
        print(result.stdout)
        print("--- Error ---")
        print(result.stderr)
        
        if result.returncode == 0 and "VERIFICATION SUCCESS" in result.stdout:
            print(">> Verification Passed!")
        else:
            print(">> Verification Failed!")
            sys.exit(1)

if __name__ == "__main__":
    verify_system()
