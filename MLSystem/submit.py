# system/submit.py
import sys
import os
import json
import time
import uuid
import subprocess
import signal

from MLsystem.queue_manager import QueueManager
from MLsystem.utils.env_manager import EnvManager


# 外部から呼び出し可能な関数にする
def add_job(args, task_type="train", hash_id=None, target_class=None, target_member=None):
    """
    ジョブをJSONとして保存し、QueueManagerに登録する。

    args: Hydra引数リスト (train.pyに渡すもの)
    task_type: "train" または "plot"
    hash_id: 実験設定のハッシュID (required)
    target_class: Plotタスクの場合の対象クラス名 (optional)
    target_member: Plotタスクの対象メンバ (NoneならMainモデル)
    """
    queue_root = EnvManager().queue_dir
    pending_dir = os.path.join(queue_root, "pending")
    os.makedirs(pending_dir, exist_ok=True)

    # job_ハッシュid_time.time()_uuid.uuid4()
    # time.time() は浮動小数点数なので、ファイル名に使う際は少し丸めるかそのまま文字列化するか考慮が必要だが、
    # 一般的にファイル名にドットが多いと紛らわしいが、指示通り実装する。
    current_time = time.time()
    unique_id = uuid.uuid4()
    job_id = f"{hash_id}_{current_time}_{unique_id}"
    job_filename = f"job_{job_id}.json"
    job_file = os.path.join(pending_dir, job_filename)

    job_data = {
        "id": job_id,
        "hash_id": hash_id,
        "task_type": task_type,
        "submitted_at": current_time,
        "submitted_at_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time)),
        "args": args,
        "status": "pending",
    }

    if target_class:
        job_data["target_class"] = target_class
    
    if target_member:
        job_data["target_member"] = target_member

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
    pid_file = os.path.join(EnvManager().queue_dir, "runner.pid")

    if os.path.exists(pid_file):
        # 既に動いているか確認（PIDが存在してもプロセスが死んでいる場合のケアは簡易的に省略）
        return

    print("🚀 Starting background runner...")

    # コンソールウィンドウを出さずに実行（Windows用設定）
    startupinfo = None
    creationflags = 0
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        creationflags = subprocess.CREATE_NO_WINDOW

    subprocess.Popen(
        [sys.executable, "-m", "MLsystem.runner"],
        # cwd=os.getcwd(), # 廃止: EnvManager経由でパス解決するためカレントディレクトリに依存しない
        startupinfo=startupinfo,
        creationflags=creationflags,
        env=os.environ,
    )


def stop_runner():
    """実行中のRunnerを停止させる"""
    pid_file = os.path.join(EnvManager().queue_dir, "runner.pid")
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
        print("Usage: python -m MLsystem.submit [hydra arguments...]")
    else:
        add_job(args)
        # CLIからの実行時は追加してすぐに実行開始する
        ensure_runner_running()
