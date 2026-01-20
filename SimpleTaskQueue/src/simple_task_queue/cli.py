import os
import sys
import json
import time
import subprocess
import shutil
import argparse
import signal
from pathlib import Path

class TaskQueue:
    def __init__(self, queue_dir):
        self.queue_dir = Path(queue_dir)
        self.pending_dir = self.queue_dir / "pending"
        self.finished_dir = self.queue_dir / "finished"
        self.failed_dir = self.queue_dir / "failed"
        self.running_dir = self.queue_dir / "running"

        for p in [self.pending_dir, self.finished_dir, self.failed_dir, self.running_dir]:
            p.mkdir(parents=True, exist_ok=True)

        self.running = True

    def run(self):
        print(f"Starting SimpleTaskQueue worker watching {self.pending_dir}")
        print("Press Ctrl+C to stop.")
        
        while self.running:
            try:
                # Find pending jobs
                # Sort by modification time to process oldest first (FIFO)
                jobs = sorted(self.pending_dir.glob("*.json"), key=os.path.getmtime)
                
                if not jobs:
                    time.sleep(1)
                    continue

                job_file = jobs[0]
                self.process_job(job_file)

            except KeyboardInterrupt:
                print("\nStopping worker...")
                self.running = False
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(1)

    def process_job(self, job_file: Path):
        print(f"Processing job: {job_file.name}")
        
        # Move to running
        running_file = self.running_dir / job_file.name
        try:
            # Atomic move if possible, or just move
            shutil.move(str(job_file), str(running_file))
        except Exception as e:
            print(f"Failed to move job to running state: {e}")
            return

        try:
            with open(running_file, "r", encoding="utf-8") as f:
                job_data = json.load(f)
            
            cmd = job_data.get("cmd")
            cwd = job_data.get("cwd")
            
            if not cmd or not isinstance(cmd, list):
                raise ValueError("Invalid job format: 'cmd' must be a list of strings")
            
            # Run the command
            print(f"Executing: {' '.join(cmd)}")
            if cwd:
                print(f"CWD: {cwd}")
            
            env = os.environ.copy()
            # Ensure pypath_manager is respected if passed or set
            
            subprocess.run(cmd, cwd=cwd, env=env, check=True)
            
            # On success, move to finished
            finished_file = self.finished_dir / job_file.name
            shutil.move(str(running_file), str(finished_file))
            print(f"Job finished: {job_file.name}")

        except subprocess.CalledProcessError as e:
            print(f"Job failed: {job_file.name}, Return Code: {e.returncode}")
            failed_file = self.failed_dir / job_file.name
            if running_file.exists():
                shutil.move(str(running_file), str(failed_file))
        except Exception as e:
            print(f"Job execution error: {e}")
            failed_file = self.failed_dir / job_file.name
            if running_file.exists():
                shutil.move(str(running_file), str(failed_file))

def main():
    parser = argparse.ArgumentParser(description="Simple Task Queue Worker")
    parser.add_argument("--queue_dir", type=str, default="queue", help="Directory containing pending/finished/failed folders")
    args = parser.parse_args()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nCtrl+C pressed. Exiting...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    queue = TaskQueue(args.queue_dir)
    queue.run()

if __name__ == "__main__":
    main()
