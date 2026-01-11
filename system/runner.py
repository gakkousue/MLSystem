# system/runner.py
import sys
import os
import json
import time
import subprocess
import shutil
import glob
import signal
from queue_manager import QueueManager

# PIDファイル（プロセスの名札）
PID_FILE = os.path.join("queue", "runner.pid")

def setup_dirs(root):
    dirs = {
        "pending": os.path.join(root, "pending"),
        "running": os.path.join(root, "running"),
        "finished": os.path.join(root, "finished"),
        "failed": os.path.join(root, "failed"),
        "logs": os.path.join(root, "logs"),  # ログ用ディレクトリを追加
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

class Runner:
    def __init__(self):
        self.current_process = None
        self.queue_root = os.path.join(os.getcwd(), "queue")
        self.dirs = setup_dirs(self.queue_root)
        self.running = True
        self.qm = QueueManager() # QueueManagerを初期化

        # シグナル（停止命令）を受け取る設定
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)

    def handle_signal(self, signum, frame):
        """停止命令が来たら実行される"""
        print(f"🛑 Signal {signum} received. Stopping...")
        self.running = False
        
        # 子プロセス（学習）が動いていたら道連れにする
        if self.current_process and self.current_process.poll() is None:
            print("Killing current training process...")
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
        
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """終了時の後始末"""
        if os.path.exists(PID_FILE):
            try:
                os.remove(PID_FILE)
            except:
                pass

    def run(self):
        # 1. 起動時にPID（名札）を保存
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

        print(f"👷 Runner started. PID: {os.getpid()}")

        try:
            while self.running:
                # 1. QueueManagerから次のジョブIDを取得 (リスト管理)
                job_id = self.qm.pop()
                
                if not job_id:
                    # ジョブがない場合は終了
                    print("✅ No more jobs in queue list. Exiting.")
                    break

                # 2. ジョブファイルの特定
                # ファイル名は job_{id}.json と決まっている
                pending_path = os.path.join(self.dirs["pending"], f"job_{job_id}.json")
                
                if not os.path.exists(pending_path):
                    print(f"⚠️ Job file not found for ID: {job_id}")
                    continue

                self.process_job(pending_path, job_id)
                
        finally:
            self.cleanup()

    def process_job(self, job_path, job_id):
        filename = os.path.basename(job_path)
        running_path = os.path.join(self.dirs["running"], filename)
        
        # pending -> running 移動
        try:
            shutil.move(job_path, running_path)
        except FileNotFoundError:
            return # 他のプロセスが取った場合はスキップ

        with open(running_path, "r") as f:
            job_data = json.load(f)

        task_type = job_data.get("task_type", "train")
        print(f"🚀 Processing: {job_id} (Type: {task_type})")

        # 実行コマンドの分岐
        if task_type == "plot":
            # execute_plot.py をサブプロセスとして実行
            # 引数としてジョブファイルのパスを渡す
            script_path = os.path.join("system", "execute_plot.py")
            # ファイルはすでに running ディレクトリに移動されているため、running_path を渡す
            cmd = [sys.executable, script_path, running_path]
        else:
            # 通常の学習 (execute_train.py)
            cmd = [sys.executable, "system/execute_train.py"] + job_data["args"]
        
        start_time = time.time()
        
        # ログファイルのパス設定
        log_filename = f"job_{job_id}.log"
        log_path = os.path.join(self.dirs["logs"], log_filename)

        # ログファイルを開いて、標準出力・標準エラー出力を書き込む
        with open(log_path, "w", encoding="utf-8") as log_file:
            # プロセスを保持しておく（停止時に道連れにするため）
            # stdout, stderrをログファイルにリダイレクト
            self.current_process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            
            # 終了待ち
            return_code = self.current_process.wait()
        
        duration = time.time() - start_time
        self.current_process = None # 終わったらクリア

        # 結果移動
        if return_code == 0:
            dest = os.path.join(self.dirs["finished"], filename)
            status = "finished"
            error_msg = None
        else:
            dest = os.path.join(self.dirs["failed"], filename)
            status = "failed"
            # 失敗時はログの最後の方を読み取ってエラーメッセージとして取得する
            error_msg = self._tail_log(log_path)

        shutil.move(running_path, dest)
        
        # ステータス更新
        with open(dest, "r+") as f:
            data = json.load(f)
            data["status"] = status
            data["duration"] = duration
            data["finished_at"] = time.time()
            data["log_file"] = log_path
            if error_msg:
                data["error_message"] = error_msg
            
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()

        # 掃除: finishedフォルダが溜まりすぎないように古いものを削除 (最新20件保持)
        if status == "finished":
            self.cleanup_old_jobs(self.dirs["finished"], keep_limit=20)
            # 成功した場合、古いログファイルも掃除しても良いが、今回は残す方針とする
            
    def _tail_log(self, path, lines=20):
        """ログファイルの末尾を取得するヘルパー"""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                # 簡易的な実装: ファイルサイズが小さければ全部読む
                content = f.readlines()
                return "".join(content[-lines:])
        except Exception:
            return "Could not read log file."

    def cleanup_old_jobs(self, target_dir, keep_limit=20):
        """指定フォルダ内のJSONファイルが多すぎる場合、古い順に削除する"""
        try:
            files = glob.glob(os.path.join(target_dir, "*.json"))
            if len(files) <= keep_limit:
                return

            # 更新日時が古い順にソート
            files.sort(key=os.path.getmtime)
            
            # 削除対象: 全体数 - 残す数
            num_to_delete = len(files) - keep_limit
            
            for f in files[:num_to_delete]:
                try:
                    os.remove(f)
                    print(f"🧹 Auto-cleaned old log: {os.path.basename(f)}")
                except Exception as e:
                    print(f"⚠️ Failed to delete {f}: {e}")
        except Exception:
            pass

if __name__ == "__main__":
    runner = Runner()
    runner.run()