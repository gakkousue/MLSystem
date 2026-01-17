import sys
import os
from MLsystem.utils.env_manager import EnvManager

# プロジェクトルート（definitionsがある階層）をsys.pathに追加して
# ユーザー定義のモデル等を from definitions... でインポートできるようにする
_env = EnvManager()
_proj_root = os.path.dirname(_env.registry_path)
if _proj_root not in sys.path:
    # 優先度を上げるため先頭付近（0はスクリプトのディレクトリなので1あたり）に挿入推奨だが、
    # appendでも基本的には動作する。安全のためappendにするが、
    # 既存の同名モジュールとの競合を避けるなら insert(0, ...) の検討も必要。
    # ここでは既存に従い、またわかりやすさ優先で append とする。
    sys.path.append(_proj_root)
