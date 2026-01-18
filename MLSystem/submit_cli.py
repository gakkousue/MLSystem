# system/submit_cli.py
import sys
import os
import json
import argparse
from pprint import pprint

from MLsystem.utils.hydra_helper import dict_to_hydra_args
from MLsystem.inspector import get_available_plots
from MLsystem.submit import add_job, calculate_hash


def extract_names_from_args(args):
    model = next((a.split("=")[1] for a in args if a.startswith("model=")), None)
    adapter = next((a.split("=")[1] for a in args if a.startswith("adapter=")), None)
    dataset = next((a.split("=")[1] for a in args if a.startswith("dataset=")), None)
    return model, adapter, dataset


def main():
    parser = argparse.ArgumentParser(
        description="Submit a job (Train/Plot) interactively from config/job file."
    )
    parser.add_argument(
        "file", help="Path to 'config.json' or 'job_*.json'"
    )
    args = parser.parse_args()

    filepath = args.file
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    print(f">> Loading file: {filepath}")
    
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)

    # 1. Hydra引数の復元
    hydra_args = []
    
    if "args" in data and isinstance(data["args"], list):
        print(">> Detected 'job.json' format.")
        hydra_args = data["args"]
    elif "model" in data and "dataset" in data:
        print(">> Detected 'config.json' format.")
        hydra_args = dict_to_hydra_args(data)
    else:
        print("Error: Unknown JSON format. Expected config.json or job.json.")
        sys.exit(1)

    # 2. 現在の設定表示
    model_name, adapter_name, dataset_name = extract_names_from_args(hydra_args)
    print("-" * 40)
    print(f"Model:   {model_name}")
    print(f"Adapter: {adapter_name}")
    print(f"Dataset: {dataset_name}")
    
    # ハッシュIDの仮計算と表示
    current_hash = calculate_hash(hydra_args)
    print(f"Hash ID: {current_hash}")
    print("-" * 40)

    # 3. タスク選択
    while True:
        mode = input("Select Task [1: Train, 2: Plot, q: Quit]: ").strip().lower()
        if mode == "q":
            sys.exit(0)
        if mode in ["1", "train"]:
            task_type = "train"
            target_class = None
            target_member = None
            break
        if mode in ["2", "plot"]:
            task_type = "plot"
            break
        print("Invalid selection.")

    # 4. Plot詳細選択
    if task_type == "plot":
        print("\n>> Scanning available plots...")
        plots = get_available_plots(model_name, adapter_name, dataset_name)
        
        if not plots:
            print("No plots available for this configuration.")
            sys.exit(1)
            
        print("Available Plots:")
        for i, p in enumerate(plots):
            print(f"  [{i+1}] {p['label']}")
            
        while True:
            sel = input("\nSelect Plot Number (q: Quit): ").strip()
            if sel.lower() == "q":
                sys.exit(0)
            try:
                idx = int(sel) - 1
                if 0 <= idx < len(plots):
                    selected = plots[idx]
                    target_class = selected["class"].__name__
                    target_member = selected["target"]
                    print(f">> Selected: {selected['label']}")
                    break
                else:
                    print("Index out of range.")
            except ValueError:
                print("Invalid input.")

    # 5. 追加パラメータ
    print("\n>> Add extra overrides? (e.g. 'common.max_epochs=5')")
    print("   Leave empty to finish.")
    while True:
        line = input("Override > ").strip()
        if not line:
            break
        if not line.startswith("+") and not line.startswith("++"):
            # 簡易追加
            hydra_args.append(f"+{line}")
        else:
            hydra_args.append(line)

    # 6. 確認と実行
    # パラメータ変更後のハッシュ再計算
    final_hash = calculate_hash(hydra_args)
    
    print("\n" + "=" * 40)
    print(f"Task Type: {task_type}")
    print(f"Hash ID:   {final_hash}")
    if task_type == "plot":
        print(f"Target Class: {target_class}")
        if target_member:
            print(f"Target Member: {target_member}")
    print("Final Arguments:")
    pprint(hydra_args)
    print("=" * 40)
    
    confirm = input("Submit this job? [y/N]: ").strip().lower()
    if confirm == "y":
        # add_job には hash_id を渡さず、内部で再計算させる（または計算済みの値を渡してもよいが、設計に従いNoneを渡す）
        # ただしここでは計算済みなので、無駄を省くなら渡しても良い。
        # 指示に従い「submit.pyにハッシュidを渡してはいけません」= add_job内部計算、とする。
        add_job(
            hydra_args, 
            task_type=task_type, 
            hash_id=None,  # 自動計算
            target_class=target_class, 
            target_member=target_member
        )
    else:
        print("Cancelled.")

if __name__ == "__main__":
    main()