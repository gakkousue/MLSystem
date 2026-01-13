# system/execute_train.py
import os
import sys
import json
import yaml

# プロジェクトルートにパスを通す
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from system.registry import Registry
from system.cli import CustomLightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint

def convert_job_to_yaml(job_data, registry):
    """
    既存のJob JSON (短縮名) を LightningCLI用の YAML 構造に変換する。
    Registryを使用して完全なクラスパスを解決する。
    """
    # 1. 引数の解析 (Hydra形式のリスト "key=value" を辞書に変換)
    # 簡易的なパース処理
    args_dict = {}
    raw_args = job_data.get("args", [])
    
    # argsから model=xxx, adapter=yyy, dataset=zzz を抽出
    base_conf = {"model": None, "adapter": None, "dataset": None}
    
    # パラメータ用辞書
    params = {
        "common": {},
        "model_params": {},
        "adapter_params": {},
        "data_params": {}
    }

    for arg in raw_args:
        if "=" not in arg: continue
        key, val = arg.split("=", 1)
        
        # 基本設定
        if key in base_conf:
            base_conf[key] = val
        # 各種パラメータ (+common.max_epochs=10 等)
        elif key.startswith("+"):
            # +common.max_epochs -> common, max_epochs
            group, param = key[1:].split(".", 1)
            # 値の型変換 (簡易)
            try:
                if val.lower() == "true": val = True
                elif val.lower() == "false": val = False
                elif "." in val: val = float(val)
                else: val = int(val)
            except: pass # 文字列のまま
            
            if group in params:
                params[group][param] = val

    # 2. Registryから情報を取得
    model_name = base_conf["model"]
    adapter_name = base_conf["adapter"]
    dataset_name = base_conf["dataset"]
    
    model_info = registry.get_model_info(model_name)
    adapter_info = registry.get_adapter_info(model_name, adapter_name)
    dataset_info = registry.get_dataset_info(dataset_name)
    
    # 3. クラスパスの生成
    # registry.jsonの情報からモジュールパスを推測して結合
    def get_class_path(info, file_key, class_name):
        base = info["base_dir"].replace("/", ".")
        file_mod = info[file_key].replace(".py", "")
        return f"{base}.{file_mod}.{class_name}"
        
    def get_module_path(info, file_key):
        base = info["base_dir"].replace("/", ".")
        file_mod = info[file_key].replace(".py", "")
        return f"{base}.{file_mod}"

    # 4. YAML構造の構築
    # パラメータのマージ順序: common < specific
    
    # Data
    data_args = params["common"].copy() # common (batch_size等)
    data_args.update(params["data_params"])
    
    # Model
    model_args = params["model_params"] # model固有
    
    # Adapter
    adapter_args = params["adapter_params"]

    yaml_config = {
        "trainer": {
            "max_epochs": params["common"].get("max_epochs", 10),
            "accelerator": "auto",
            "devices": 1,
            # Logger設定などは後で注入も可能だがここではシンプルに
            "default_root_dir": f"output/experiments/{job_data.get('id', 'temp')}" 
        },
        "model": {
            "class_path": get_class_path(model_info, "main_file", "Model"),
            "init_args": model_args
        },
        "data": {
            "class_path": get_class_path(dataset_info, "main_file", "DataModule"),
            "init_args": data_args
        },
        "adapter": {
            "module_path": get_module_path(adapter_info, "main_file"),
            "init_args": adapter_args
        }
    }
    
    return yaml_config

def main():
    # 引数からジョブファイルのパスを取得
    # system/submit.py は "system/execute_train.py" + args を呼ぶが、
    # Runner経由で呼ばれる場合はJSONパスが渡される想定に変更する必要があるかも。
    # しかし今回はRunnerがJSONを解析して execute_train.py を呼ぶ方式のまま、
    # Runnerが "引数リスト" ではなく "JSONパス" を渡すように runner.py を修正するか、
    # ここでHydra引数形式をパースするかの二択。
    # ユーザー指示は「yamlファイルを作成して読み込む」なので、
    # ここでは便宜上「Hydra引数形式の引数」を受け取って、内部でYAML化してCLIに渡すフローにする。
    
    # Runnerからの呼び出し引数互換性のため、sys.argvを解析
    # Runnerは `[sys.executable, "system/execute_train.py"] + job_data["args"]` で呼んでいる
    # job_data["args"] は ["model=resnet", "+common.max_epochs=10", ...] というリスト
    
    # 簡易的なJobデータ復元
    dummy_job_data = {
        "id": "manual_run", # 実際はハッシュID等が欲しいが、引数からは復元困難な場合がある
        "args": sys.argv[1:]
    }
    
    registry = Registry()
    
    # 1. 設定YAMLの生成
    config_dict = convert_job_to_yaml(dummy_job_data, registry)
    
    # 実験ID (Hash) の特定 (config_diff.jsonがあればそれを使う、なければ生成)
    # ここでは簡易的に output/experiments/temp_cli に保存設定
    # ※ 本来はHashingロジックを通すべきだが、CLI化に伴い構成が変わるため一時的なパスを使用
    
    # 一時ファイルとしてYAML保存
    os.makedirs("temp_configs", exist_ok=True)
    config_path = "temp_configs/run_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)
        
    print(f">> Generated CLI Config: {config_path}")

    # 2. 動的な設定の注入 (Callbacks等)
    # YAMLファイルには書かず、CLI実行時の引数や変数で注入する例
    # ModelCheckpointは頻繁に使うので、常に有効化する設定を動的に生成
    
    checkpoint_callback = ModelCheckpoint(
        filename="epoch={epoch:02d}-{val_acc:.2f}",
        save_last=True,
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )
    
    # 3. CustomLightningCLI の実行
    # args引数に ["--config", config_path] を渡すことでファイルから設定をロードさせる
    # run=True で即座に学習開始
    
    cli = CustomLightningCLI(
        args=["--config", config_path],
        run=False, # 一旦Falseにして設定後にfit
        save_config_callback=None, # config保存は自前で行うかLightningに任せる
        subclass_mode_model=True, # modelセクションでclass_path指定を有効化
        subclass_mode_data=True,  # dataセクションでclass_path指定を有効化
    )
    
    # Callbackの注入 (Trainerの設定を直接操作)
    cli.trainer.callbacks.append(checkpoint_callback)
    
    print(">> Starting Training with CustomLightningCLI...")
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
    
    # 完了マーカー
    # CLIはデフォルトでログディレクトリを作るが、Runnerが期待する done ファイル等は別途必要ならここで作成
    
if __name__ == "__main__":
    main()