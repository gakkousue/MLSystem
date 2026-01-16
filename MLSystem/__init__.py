import os
import sys

# プロジェクトルート（カレントディレクトリ）をsys.pathに追加
# これにより、MLsystemパッケージ内から common や definitions をインポート可能にする
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
