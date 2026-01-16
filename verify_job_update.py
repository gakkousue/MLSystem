
import sys
import os
import json
import time
import shutil
import glob

# Ensure MLSystem is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MLsystem.submit import add_job
from MLsystem.utils.env_manager import EnvManager

def verify():
    print("--- Starting Verification ---")
    
    # 1. Submit a Job
    hash_id = "test_hash_123"
    args = ["param1=10", "param2=20"]
    target_class = "TestPlotClass"
    
    print("\n1. submitting job...")
    job_id = add_job(args, task_type="plot", hash_id=hash_id, target_class=target_class)
    print(f"Job ID: {job_id}")
    
    # 2. Check File Name Format
    # Format: hash_id_timestamp_uuid
    parts = job_id.split('_')
    if len(parts) < 3:
         print(f"❌ Job ID format incorrect: {job_id}")
    else:
         print(f"✅ Job ID format looks correct (parts: {len(parts)})")
         
    if not job_id.startswith(hash_id):
         print(f"❌ Job ID should start with hash_id: {job_id}")
    else:
         print(f"✅ Job ID starts with hash_id")

    # 3. Check JSON Content (Pending)
    queue_dir = EnvManager().queue_dir
    pending_file = os.path.join(queue_dir, "pending", f"job_{job_id}.json")
    
    if not os.path.exists(pending_file):
        print(f"❌ Job file not created: {pending_file}")
        return

    with open(pending_file, 'r') as f:
        data = json.load(f)
        
    print("\n--- JSON Content (Pending) ---")
    print(json.dumps(data, indent=2))
    
    if data.get("hash_id") != hash_id:
        print("❌ hash_id missing or incorrect")
    else:
        print("✅ hash_id correct")
        
    if data.get("target_class") != target_class:
        print("❌ target_class missing or incorrect")
    else:
        print("✅ target_class correct")
        
    if "condition" in data:
        print("❌ condition field should be removed")
    else:
        print("✅ condition field is absent")

    # 4. Simulate Runner Execution Logic
    print("\n2. Simulating Runner execution...")
    # Move to running
    running_file = os.path.join(queue_dir, "running", f"job_{job_id}.json")
    os.makedirs(os.path.join(queue_dir, "running"), exist_ok=True)
    shutil.move(pending_file, running_file)
    
    # Simulate processing
    start_time = time.time()
    time.sleep(1.5) # Wait a bit
    
    # Finished
    duration = time.time() - start_time
    
    finished_dir = os.path.join(queue_dir, "finished")
    os.makedirs(finished_dir, exist_ok=True)
    dest_file = os.path.join(finished_dir, f"job_{job_id}.json")
    
    shutil.move(running_file, dest_file)
    
    # Write status and times (mimicking runner.py logic)
    with open(dest_file, "r+") as f:
        data = json.load(f)
        data["status"] = "finished"
        data["started_at"] = start_time
        data["finished_at"] = time.time()
        data["duration"] = duration
        
        m, s = divmod(duration, 60)
        h, m = divmod(m, 60)
        data["duration_human"] = "{:d}:{:02d}:{:02d}".format(int(h), int(m), int(s))
        
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
        
    # 5. Check Final JSON
    with open(dest_file, 'r') as f:
        final_data = json.load(f)
        
    print("\n--- JSON Content (Finished) ---")
    print(json.dumps(final_data, indent=2))
    
    if "started_at" not in final_data:
        print("❌ started_at missing")
    else:
        print("✅ started_at present")

    if "started_at_str" not in final_data:
        print("❌ started_at_str missing")
    else:
        print(f"✅ started_at_str present: {final_data['started_at_str']}")
    
    if "finished_at_str" not in final_data:
        print("❌ finished_at_str missing")
    else:
        print(f"✅ finished_at_str present: {final_data['finished_at_str']}")
        
    if "duration_human" not in final_data:
        print("❌ duration_human missing")
    else:
        print(f"✅ duration_human present: {final_data['duration_human']}")
    
    # Cleanup
    try:
        os.remove(dest_file)
        print("\nCleanup done.")
    except:
        pass

if __name__ == "__main__":
    verify()
