# MLSystem 設計書 (Core Framework) Ver 5.0

## 1. システム概要
本システムは、Hydraの標準機能を最大限に活用した、疎結合かつ拡張性の高い深層学習実験管理フレームワークである。

### コア・コンセプト
1.  **Hydra Native Architecture**
    *   `hydra.utils.instantiate` を使用して DataModule と Model を構築。
    *   設定は YAML 形式で管理し、Defaults List を活用する。
2.  **Decoupled Definitions**
    *   Dataset と Model 間の直接的な依存を排除。
    *   明示的な引数渡しによるインスタンス化。
3.  **Explicit Resume & Reference**
    *   `source_run_dir` を指定することで、過去の実験結果を再利用。

## 2. ディレクトリ構成
```text
MLSystem/
├── configs/
│   ├── config.yaml           # Hydraルート設定
│   ├── dataset/              # 各データセットの設定
│   └── model/                # 各モデルの設定
├── execute_train.py          # 学習実行スクリプト
└── execute_plot.py           # 可視化実行スクリプト

definitions/
├── datasets/                 # データセット定義 (DataModule + Loader)
└── models/                   # モデル定義 (LightningModule)
```

## 3. 実行パイプライン

### 学習 (Train)
*   **コマンド**: `python MLSystem/execute_train.py dataset=... model=...`
*   **特徴**:
    *   `experiments/{exp_name}/latest` シンボリックリンクを自動作成。
    *   `source_run_dir` 指定によりチェックポイントからの継続学習が可能。

### 可視化 (Plot)
*   **コマンド**: `python MLSystem/execute_plot.py source_run_dir=...`
*   **特徴**:
    *   指定したディレクトリの `.hydra/config.yaml` を読み込み、学習時の環境を忠実に復元。

## 4. 関連パッケージ
本システムは以下の独立パッケージを利用する。
*   [PyPathManager](設計書_PyPathManager.md): 環境変数によるパス解決。
*   [SimpleTaskQueue](設計書_SimpleTaskQueue.md): ジョブのキュー管理と実行。
