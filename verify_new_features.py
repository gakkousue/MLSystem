
import os
import sys
import json
import shutil
import unittest
from dataclasses import dataclass, field
from typing import Any

# パス設定のモック
original_registry_path = os.environ.get("MLSYSTEM_REGISTRY_PATH")

test_dir = os.path.abspath("test_registry_env")
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
os.makedirs(test_dir)

registry_path = os.path.join(test_dir, "registry.json")
os.environ["MLSYSTEM_REGISTRY_PATH"] = registry_path

# ダミーファイル作成
os.makedirs(os.path.join(test_dir, "base_dir_A"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "base_dir_B"), exist_ok=True)
with open(os.path.join(test_dir, "base_dir_A", "plot_A.py"), "w") as f:
    f.write('''
from MLsystem.utils.base_plot import BasePlot
class PlotA(BasePlot):
    pass
''')
with open(os.path.join(test_dir, "base_dir_B", "plot_B.py"), "w") as f:
    f.write('''
from MLsystem.utils.base_plot import BasePlot
class PlotB(BasePlot):
    TITLE = "Custom Title B"
    pass
''')

# Registryデータ
registry_data = {
    "project_root": ".",
    "models": {
        "ParentModel": {
            "base_dir": "base_dir_A",
            "plot_file": "plot_A.py",
            "adapters": {
                "ad_A": {"base_dir": "adapters/ad_A", "config_file": "conf.py"}
            }
        },
        "ChildModel": {
            "parent": "ParentModel",
            "base_dir": "base_dir_B", # 上書き
            # plot_file は未定義 -> 親から継承されるはずだがパスはどうなる？
            # 親: base_dir_A/plot_A.py (絶対パス化される)
            "components": {
                "sub": "MemberModel"
            }
        },
        "MemberModel": {
            "base_dir": "base_dir_B",
            "plot_file": "plot_B.py"
        }
    }
}
with open(registry_path, "w") as f:
    json.dump(registry_data, f)

# モジュールインポート (環境変数設定後に)
import MLsystem.registry as reg_module
import MLsystem.inspector as insp_module
from MLsystem.utils.config_base import conf_field, BaseConfig
from MLsystem.hashing import compute_combined_hash

class TestMLSystemFeatures(unittest.TestCase):
    def test_registry_inheritance(self):
        reg = reg_module.Registry()
        
        # 親情報の取得
        parent = reg.get_model_info("ParentModel")
        self.assertTrue(parent["plot_file"].endswith("plot_A.py"))
        
        # 子情報の取得 (継承確認)
        child = reg.get_model_info("ChildModel")
        
        # 親の plot_file が引き継がれているか
        # かつ、親の base_dir ("base_dir_A") 基準で絶対パスになっているか
        self.assertIn("plot_file", child)
        self.assertTrue(os.path.isabs(child["plot_file"]))
        self.assertIn("base_dir_A", child["plot_file"]) # 親のディレクトリ
        
        # 子自身の base_dir ("base_dir_B") は絶対パス化されているか
        self.assertIn("base_dir_B", child["base_dir"])
        
        # Adapters継承
        self.assertIn("adapters", child)
        self.assertIn("ad_A", child["adapters"])
        # Adapterのbase_dirパスチェック
        # 親のbase_dir_A/adapters/ad_A になるべき
        # (ロジック修正により、親のadaptersも親base_dir基準で生成される)
        ad_info = reg.get_adapter_info("ChildModel", "ad_A")
        self.assertIn("base_dir_A", ad_info["base_dir"])

    def test_inspector_collection(self):
        # Plotファイルのパス解決とロードができるか
        # 実際にファイルを作ったのでロードできるはず
        
        plots = insp_module.get_available_plots("ChildModel")
        
        # 期待されるPlot:
        # 1. ParentModel由来 (PlotA) - Label: "[Model] PlotA" (継承により自分自身のPlot扱い)
        #    ※子でplot_fileを定義していないので親のfileが採用され、それがMain扱いになる
        # 2. ParentModel (Parent) - 親を辿るロジックで再度収集される可能性がある
        #    ※ParentModelはChildModelの親なので、Parent(ParentModel)として収集される？
        #    現在の実装では、まずMainのPlotをとり、その後親を辿る。
        #    MainのPlotは親由来のもの。その後親を辿ると同じファイルを見る。
        #    => 重複する可能性があるが仕様としてはOK
        # 3. MemberModel (Component) - Label: "[Comp.sub] Custom Title B"
        
        labels = [p["label"] for p in plots]
        targets = [p["target"] for p in plots]
        
        print("Collected Plots:", labels)
        
        # Main (継承)
        has_plot_a = any("PlotA" in l for l in labels)
        self.assertTrue(has_plot_a)
        
        # Component
        has_plot_b = any("Custom Title B" in l and "Comp.sub" in l for l in labels)
        self.assertTrue(has_plot_b)
        
        # Target check
        for p in plots:
            if "Comp.sub" in p["label"]:
                self.assertEqual(p["target"], "sub")

    def test_config_excludes(self):
        @dataclass
        class MyConf(BaseConfig):
            normal: str = "val"
            # excludes付与。可変デフォルト値エラーを避けるためNoneをデフォルトにし、
            # 実運用でもあり得るパターンとして扱うか、あるいは単純に修正する。
            # conf_fieldの実装が field(default=default, ...) となっているので、
            # 辞書を直接渡すとエラーになる。
            complex_dict: dict = conf_field(default=None, excludes=["ignored_key"])
        
        c = MyConf()
        c.complex_dict = {"keep": 1, "ignored_key": 999} # 手動セット
        
        # param 1: ignored_key あり
        params1 = {
            "_name": "test",
            "normal": "val",
            "complex_dict": {"keep": 1, "ignored_key": 999}
        }
        
        # param 2: ignored_key 違い (ハッシュ同じになるべき)
        params2 = {
            "_name": "test",
            "normal": "val",
            "complex_dict": {"keep": 1, "ignored_key": 111}
        }
        
        # param 3: keep 違い (ハッシュ変わるべき)
        params3 = {
            "_name": "test",
            "normal": "val",
            "complex_dict": {"keep": 2, "ignored_key": 999}
        }
        
        h1, _ = compute_combined_hash(None, {}, MyConf, params1, None, {}, None, {})
        h2, _ = compute_combined_hash(None, {}, MyConf, params2, None, {}, None, {})
        h3, _ = compute_combined_hash(None, {}, MyConf, params3, None, {}, None, {})
        
        self.assertEqual(h1, h2, "Excludes keys should be ignored in hash")
        self.assertNotEqual(h1, h3, "Other keys should affect hash")

if __name__ == "__main__":
    unittest.main()
