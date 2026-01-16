# system/env_setup.py
"""
環境変数 MLSYSTEM_SYSTEM_PATH を設定するユーティリティモジュール

このモジュールをインポートすることで、環境変数が自動的に設定されます。
エントリーポイントやsystem内のモジュールから使用します。
"""
import os
import sys


def setup_environment():
  """
  環境変数 MLSYSTEM_SYSTEM_PATH を設定し、sys.pathに必要なパスを追加します。
  
  既に環境変数が設定されている場合はスキップします。
  """
  # 環境変数が既に設定されている場合はスキップ
  if 'MLSYSTEM_SYSTEM_PATH' in os.environ:
    return
  
  # このファイル（env_setup.py）の絶対パスからsystemフォルダのパスを取得
  current_file = os.path.abspath(__file__)
  system_path = os.path.dirname(current_file)
  
  # systemフォルダのパスを環境変数に設定
  os.environ['MLSYSTEM_SYSTEM_PATH'] = system_path


def get_system_path():
  """
  環境変数からsystemフォルダの絶対パスを取得します。
  
  Returns:
    str: systemフォルダの絶対パス
    
  Raises:
    RuntimeError: 環境変数が設定されていない場合
  """
  if 'MLSYSTEM_SYSTEM_PATH' not in os.environ:
    raise RuntimeError(
      "環境変数 MLSYSTEM_SYSTEM_PATH が設定されていません。"
      "env_setup.setup_environment() を先に呼び出してください。"
    )
  
  system_path = os.environ['MLSYSTEM_SYSTEM_PATH']
  
  # パスの存在チェック
  if not os.path.exists(system_path):
    raise FileNotFoundError(f"systemフォルダが見つかりません: {system_path}")
  
  return system_path


def get_project_root():
  """
  systemフォルダの親ディレクトリ（プロジェクトルート）を取得します。
  
  Returns:
    str: プロジェクトルートの絶対パス
  """
  system_path = get_system_path()
  project_root = os.path.dirname(system_path)
  
  # パスの存在チェック
  if not os.path.exists(project_root):
    raise FileNotFoundError(f"プロジェクトルートが見つかりません: {project_root}")
  
  return project_root


def add_to_sys_path():
  """
  systemフォルダとプロジェクトルートをsys.pathに追加します。
  既に追加されている場合はスキップします。
  """
  system_path = get_system_path()
  project_root = get_project_root()
  
  # systemフォルダをsys.pathに追加（system内モジュールのインポート用）
  if system_path not in sys.path:
    sys.path.insert(0, system_path)
  
  # プロジェクトルートをsys.pathに追加（commonやdefinitionsのインポート用）
  if project_root not in sys.path:
    sys.path.insert(0, project_root)


# モジュールがインポートされたときに自動的に環境変数を設定
setup_environment()
