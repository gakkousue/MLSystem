# system/queue_manager.py
import os
import json
import time

QUEUE_DIR = "queue"
LIST_FILE = os.path.join(QUEUE_DIR, "list.json")
LOCK_DIR = os.path.join(QUEUE_DIR, "queue.lock")

class FileLock:
    """
    ディレクトリ作成の原子性を利用した簡易ファイルロッククラス。
    複数のプロセス（GUI, Runner）が同時にリストを書き換えるのを防ぐ。
    """
    def __init__(self, lock_path, timeout=10):
        self.lock_path = lock_path
        self.timeout = timeout

    def __enter__(self):
        start_time = time.time()
        while True:
            try:
                # mkdirは原子的な操作なので、成功すればロック獲得とみなせる
                os.mkdir(self.lock_path)
                return True
            except FileExistsError:
                # 既に誰かがロックしている場合
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Could not acquire lock: {self.lock_path}")
                time.sleep(0.05)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            os.rmdir(self.lock_path)
        except OSError:
            pass

class QueueManager:
    """
    実行待ちジョブのIDリストを管理するクラス。
    queue/list.json を読み書きする。
    """
    def __init__(self):
        os.makedirs(QUEUE_DIR, exist_ok=True)
        # リストファイルがなければ初期化
        if not os.path.exists(LIST_FILE):
            with open(LIST_FILE, "w") as f:
                json.dump([], f)

    def _load_list(self):
        try:
            with open(LIST_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_list(self, data):
        with open(LIST_FILE, "w") as f:
            json.dump(data, f, indent=4)

    def push(self, job_id):
        """末尾にジョブを追加する（通常の登録）"""
        with FileLock(LOCK_DIR):
            q = self._load_list()
            if job_id not in q:
                q.append(job_id)
                self._save_list(q)

    def insert_front(self, job_id):
        """先頭にジョブを割り込ませる（優先実行や、不足分の学習用）"""
        with FileLock(LOCK_DIR):
            q = self._load_list()
            # 既に存在する場合は位置を移動させる
            if job_id in q:
                q.remove(job_id)
            q.insert(0, job_id)
            self._save_list(q)

    def pop(self):
        """先頭のジョブIDを取得してリストから削除する（Runnerが使用）"""
        with FileLock(LOCK_DIR):
            q = self._load_list()
            if not q:
                return None
            job_id = q.pop(0)
            self._save_list(q)
            return job_id

    def remove(self, job_id):
        """指定したジョブIDをリストから削除する（キャンセル時など）"""
        with FileLock(LOCK_DIR):
            q = self._load_list()
            if job_id in q:
                q.remove(job_id)
                self._save_list(q)
                
    def get_list(self):
        """現在のキューリストを取得（読み取り専用、GUI表示用）"""
        # 読み取りだけでもロックを一瞬かけることで、書き込み中の不完全な状態を読むのを防ぐ
        with FileLock(LOCK_DIR):
            return self._load_list()