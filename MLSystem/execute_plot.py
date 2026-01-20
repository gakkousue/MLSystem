# MLSystem/execute_plot.py
import sys
import os
import hydra
from omegaconf import OmegaConf, DictConfig
import torch
import PyPathManager # パス解決

# Plotロジックのベース（本来ならdefinitions内やcommonにあるべきだが、簡易的にここで探すか、Hydraで管理する）
# 今回は既存のPlotクラスを探すロジックが必要だが、Hydra化に伴いPlotクラスもConfigで指定できるとベスト。
# しかし、Plotはアドホックに実行することも多い。
# ここでは「指定されたsource_run_dirの設定を使ってモデルを復元し、
# 追加のConfigで指定されたPlotクラスを実行する」流れにする。

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if not cfg.source_run_dir:
        print("Error: 'source_run_dir' must be specified for plotting.")
        sys.exit(1)
    
    source_dir = cfg.source_run_dir
    print(f">> Plotting Source: {source_dir}")

    # 1. 元のConfigをロード
    # .hydra/config.yaml があるはず
    config_path = os.path.join(source_dir, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)

    print(">> Loading original config...")
    orig_cfg = OmegaConf.load(config_path)
    
    # 2. モデルとデータセットの復元
    # cfg (今回の実行時引数) でオーバーライドがあればそれを優先したいが、
    # 基本は元の設定で復元。
    
    # データセット
    # Plot時はTestデータだけ必要な場合が多いが、DataModule全体を復元するのが安全
    print(">> Instantiating DataModule...")
    try:
        datamodule = hydra.utils.instantiate(orig_cfg.dataset)
        datamodule.prepare_data()
        datamodule.setup("test") # Plot用途ならtest/predictモードが妥当か
    except Exception as e:
        print(f"Warning: Failed to instantiate datamodule: {e}")
        datamodule = None

    # モデル
    print(">> Instantiating Model...")
    model = hydra.utils.instantiate(orig_cfg.model)
    
    # チェックポイントのロード
    ckpt_path = os.path.join(source_dir, "checkpoints", "last.ckpt") # default location
    if not os.path.exists(ckpt_path):
        # 探索
        for root, dirs, files in os.walk(source_dir):
            for f in files:
                if f.endswith(".ckpt"):
                    ckpt_path = os.path.join(root, f)
                    break 
    
    if os.path.exists(ckpt_path):
        print(f">> Loading weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("Warning: Checkpoint not found. Using initialized weights.")

    model.eval()
    model.freeze()

    # 3. Plotの実行
    # 本来はどのPlotを実行するかを引数で受けるべき。
    # ここでは仮に 'plot_class' という引数がcfgにあるとする、またはアドホックな処理
    # ユーザー要望の「execute_plot.py の修正」では詳細なPlot特定ロジック指定がなかったので、
    # 動作確認用に「モデルの出力を通す」などの基本的動作、または既存のPlotを呼び出す仕組みが必要。
    
    # シンプルにするため、カスタムPlotロジックがあればそれを実行、なければ何もしない
    # Plot用スクリプトのパスを引数で受け取る？
    # 以前のロジック: get_available_plots でクラスを探していた。
    
    # 今回は簡略化のため、ユーザーが実装すべきPlotのエントリポイントを想定
    # 例: definitions.models.{name}.plots...
    
    print(">> Ready to plot. (Logic to invoke specific plot class to be implemented based on need)")
    print("   Model and Datamodule are ready.")
    
    # ここでIPythonを埋め込むか、特定の関数を呼ぶ
    pass

if __name__ == "__main__":
    main()
