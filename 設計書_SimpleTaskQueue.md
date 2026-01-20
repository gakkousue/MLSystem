# SimpleTaskQueue 設計書

## 1. 概要
特定のディレクトリを監視し、JSON 形式で定義されたジョブ（コマンド）を順次実行するための軽量タスクキューシステム。
機械学習の学習ジョブや可視化ジョブをバックグラウンドで並列性を制御しながら実行することを目的とする。

## 2. 状態管理（ディレクトリ構造）
*   `pending/`: 実行待ちのジョブ定義 (JSON)。
*   `running/`: 現在実行中のジョブ。
*   `finished/`: 正常終了したジョブ。
*   `failed/`: エラー終了したジョブ。

## 3. ジョブ定義 (JSON) の形式
```json
{
    "cmd": ["python", "my_script.py", "--arg1", "val1"],
    "cwd": "c:/work/project"
}
```

## 4. 実行方法
ワーカー（実行監視プロセス）を起動する。
```bash
python -m simple_task_queue --queue_dir path/to/queue
```

## 5. 機能仕様
*   FIFO (First-In First-Out) 方式で古いジョブから順に実行。
*   `subprocess` を介してコマンドを実行。
*   実行成功/失敗に応じてジョブファイルを `finished/` または `failed/` へ自動移動。
*   外部依存のない単一の Python プロセスとして動作。
