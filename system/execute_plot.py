# system/execute_plot.py
import sys
import os
import json
import traceback

# プロジェクトルートにパスを通す
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from system.loader import ExperimentLoader
from system.inspector import get_available_plots

def main():
    if len(sys.argv) < 2:
        print("Usage: python system/execute_plot.py <job_json_path>")
        sys.exit(1)

    job_path = sys.argv[1]
    
    print(f">> Starting Plot Execution. Job: {job_path}")
    
    # 1. Job定義の読み込み
    try:
        with open(job_path, "r") as f:
            job_data = json.load(f)
    except Exception as e:
        print(f"Error loading job file: {e}")
        sys.exit(1)
        
    hash_id = job_data.get("hash_id")
    target_class_name = job_data.get("target_class")
    job_args = job_data.get("args", [])
    
    if not hash_id:
        print("Error: 'hash_id' is missing in job data.")
        sys.exit(1)
        
    if not target_class_name:
        print("Error: 'target_class' is missing in job data.")
        sys.exit(1)

    try:
        # 2. 実験環境の復元
        print(f">> Loading Experiment: {hash_id}")
        loader = ExperimentLoader(hash_id)
        
        # 3. Plotクラスの検索
        available_plots = get_available_plots(
            loader.model_name, 
            loader.adapter_name, 
            loader.dataset_name
        )
        
        target_cls = None
        for cls in available_plots:
            if cls.__name__ == target_class_name:
                target_cls = cls
                break
                
        if not target_cls:
            print(f"Error: Plot class '{target_class_name}' not found for model '{model_name}'.")
            print(f"Available plots: {[c.__name__ for c in available_plots]}")
            sys.exit(1)
            
        # 4. 実行
        print(f">> Executing Plot: {target_class_name}")
        
        # インスタンス化 (loaderと学習用引数を渡す)
        plot_instance = target_cls(loader, job_args)
        
        # 実行 (必要なら内部で学習が走る)
        plot_instance.run()
        
        print(">> Plot Execution Finished Successfully.")
        
    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()