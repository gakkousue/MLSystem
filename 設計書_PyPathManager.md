# PyPathManager 設計書

## 1. 概要
環境に応じて動的に Python のインポートパス (`sys.path`) を管理するための軽量ユーティリティパッケージ。
ハードコードされた絶対パスをコードから排除し、環境変数によるポータブルなパス管理を実現することを目的とする。

## 2. 機能仕様
*   環境変数 `PYPATH_CONFIG` に指定された JSON ファイルを読み込む。
*   JSON ファイル内のリストまたは辞書形式の値を `sys.path` に追加する。
*   インポート時 (`import PyPathManager`) に自動的にパスの読み込みと追加を実行する。

## 3. インストール方法
```bash
pip install -e ./PyPathManager
```

## 4. 設定ファイルの形式 (JSON)
```json
[
    "c:/absolute/path/to/my_package",
    "d:/another/path"
]
```
または
```json
{
    "alias1": "c:/absolute/path/to/my_package",
    "alias2": "d:/another/path"
}
```

## 5. 利用方法
プログラムの冒頭でインポートするだけで機能する。
```python
import PyPathManager
import my_package # 追加されたパスから読み込み可能
```
