# system/execute_plot.py
import sys
import os
import json
import traceback

from MLsystem.loader import ExperimentLoader
from MLsystem.inspector import get_available_plots


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
    target_member = job_data.get("target_member") # Optional
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

        # 3. 実行時パラメータのオーバーライド抽出 (args から +開頭のものを取得)
        from omegaconf import OmegaConf

        overrides = {}
        dot_list = [a[1:] for a in job_args if a.startswith("+")]
        if dot_list:
            try:
                # Hydra形式のドット記法を辞書に変換
                raw_overrides = OmegaConf.to_container(
                    OmegaConf.from_dotlist(dot_list), resolve=True
                )
                # loaderが期待する xxx_diff 形式に変換
                mapping = {
                    "common": "common_diff",
                    "model_params": "model_diff",
                    "adapter_params": "adapter_diff",
                    "data_params": "data_diff",
                }
                for k, v in raw_overrides.items():
                    if k in mapping:
                        overrides[mapping[k]] = v
            except Exception as e:
                print(f"[Warning] Failed to parse overrides from args: {e}")

        # オーバーライドを登録 (Plotクラス内でのsetup呼び出しにも適用されるようにする)
        loader.update_overrides(overrides)

        # モデル構築（必須）
        loader.setup()

        # 3. Plotクラスの検索
        # 戻り値: [{"class": Cls, "target": ..., "label": ...}]
        available_plots_items = get_available_plots(
            loader.model_name, loader.adapter_name, loader.dataset_name
        )

        target_cls = None
        for item in available_plots_items:
            cls = item["class"]
            if cls.__name__ == target_class_name:
                # クラス名だけで判定して良いか？
                # 同じクラス名が別コンポーネントで使われている可能性があるが、
                # execute_plot時点では「どのコンポーネント用のPlotか」は target_member で決まる。
                # ただし、target_member が一致するものを選ぶべきか？
                # GUI側では (Label) -> (Class, Target) と一意に決めている。
                # ここでは Class と Target の両方が一致するものを探すべきだが、
                # 実は Plotクラス自体はコンポーネントに依存せず定義されていることが多い。
                # 単にクラスが見つかればOKとし、Targetの適用は後述のロジックで行う。
                target_cls = cls
                break

        if not target_cls:
            print(
                f"Error: Plot class '{target_class_name}' not found for model '{loader.model_name}'."
            )
            print(f"Available plots: {[item['class'].__name__ for item in available_plots_items]}")
            sys.exit(1)

        # 4. 実行
        print(f">> Executing Plot: {target_class_name} (Target: {target_member})")

        # インスタンス化 (loaderと学習用引数を渡す)
        plot_instance = target_cls(loader, job_args)
        
        # ターゲットモデルの注入
        if target_member:
            # Mainモデルからメンバを辿る
            # 例: target_member="backbone" -> loader.model.backbone
            if not hasattr(loader.model, target_member):
                 print(f"Error: Target member '{target_member}' not found in loaded model.")
                 sys.exit(1)
            
            sub_model = getattr(loader.model, target_member)
            plot_instance.target_model = sub_model
        else:
            # 指定がなければMainモデルそのもの（またはNoneのまま BasePlot側で判断）
            plot_instance.target_model = loader.model

        # 実行 (必要なら内部で学習が走る -> 今後は廃止されエラーになる)
        plot_instance.run()

        print(">> Plot Execution Finished Successfully.")

    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
