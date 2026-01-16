# system/submit.py
import sys
import os
import json
import time
import uuid
import subprocess
import signal

# 環境変数を設定し、sys.pathに必要なパスを追加
import env_setup
env_setup.add_to_sys_path()

from queue_manager import QueueManager

# 外部から呼び出し可能な関数にする
def add_job(args, task_type="train", condition=None, extra_data=None):
  """
  ジョブをJSONとして保存し、QueueManagerに登録する。
  
  args: Hydra引数リスト (train.pyに渡すもの)
  task_type: "train" または "plot"
  condition: 実行条件 (dict, optional)
  extra_data: その他の保存したいメタデータ (dict, optional)
              例: {"hash_id": "...", "target_class": "ConfusionMatrix"}
  """
  queue_root = os.path.join(os.getcwd(), "queue")
  pending_dir = os.path.join(queue_root, "pending")
  os.makedirs(pending_dir, exist_ok=True)

  job_id = f"{int(time.time())}_{str(uuid.uuid4())[:8]}"
  job_file = os.path.join(pending_dir, f"job_{job_id}.json")

  job_data = {
    "id": job_id,
    "task_type": task_type,
    "submitted_at": time.time(),
    "args": args,
    "condition": condition,
    "status": "pending"
  }
  
  # extra_dataがあればマージする
  if extra_data:
    job_data.update(extra_data)

  # JSONファイルとして詳細情報を保存
  with open(job_file, "w") as f:
    json.dump(job_data, f, indent=4)

  # QueueManagerを使用してリストにIDを追加
  qm = QueueManager()
  qm.push(job_id)

  print(f"✅ Job submitted! ID: {job_id} (Type: {task_type})")
  
  return job_id

def ensure_runner_running():

    # QueueManagerを使用してリストにIDを追加
    qm = QueueManager()
    qm.push(job_id)

    print(f"✅ Job submitted! ID: {job_id} (Type: {task_type})")
    
    # ここでの自動起動は廃止 (GUI等の呼び出し側で制御する)
    # ensure_runner_running()
    
    return job_id

def ensure_runner_running():
  """Runnerが動いていなければ裏で起動する"""
  pid_file = os.path.join("queue", "runner.pid")
  
  if os.path.exists(pid_file):
    # 既に動いているか確認（PIDが存在してもプロセスが死んでいる場合のケアは簡易的に省略）
    return

  print("🚀 Starting background runner...")
  
  # コンソールウィンドウを出さずに実行（Windows用設定）
  startupinfo = None
  creationflags = 0
  if os.name == 'nt':
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    creationflags = subprocess.CREATE_NO_WINDOW
    
  subprocess.Popen(
    [sys.executable, "system/runner.py"],
    cwd=os.getcwd(),
    startupinfo=startupinfo,
    creationflags=creationflags,
    env=os.environ
  )

def stop_runner():
  """実行中のRunnerを停止させる"""
  pid_file = os.path.join("queue", "runner.pid")
  if not os.path.exists(pid_file):
    print("Runner is not running.")
    return False

  try:
    with open(pid_file, "r") as f:
      pid = int(f.read().strip())
    
    # OS標準のシグナルで停止
    os.kill(pid, signal.SIGTERM)
    print(f"🛑 Runner (PID {pid}) stopped.")
    
    # PIDファイルが消えるのを少し待つ
    time.sleep(1)
    if os.path.exists(pid_file):
      try:
        os.remove(pid_file)
      except:
        pass
        
    return True
  except Exception as e:
    print(f"Failed to stop runner: {e}")
    return False

# CLIとして実行された場合
if __name__ == "__main__":
  args = sys.argv[1:]
  if not args:
    print("Usage: python system/submit.py [hydra arguments...]")
  else:
    add_job(args)
    # CLIからの実行時は追加してすぐに実行開始する
    ensure_runner_running()