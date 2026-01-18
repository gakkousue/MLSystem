import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import json
import importlib
import inspect
from dataclasses import fields, is_dataclass

# 環境変数を設定し、sys.pathに必要なパスを追加


# from MLsystem.hashing import compute_combined_hash  # 削除
from MLsystem.submit import add_job, ensure_runner_running, calculate_hash # 追加
from MLsystem.utils.hydra_helper import dict_to_hydra_args # 追加
from MLsystem.inspector import get_available_plots
from MLsystem.registry import Registry
from MLsystem.utils.env_manager import EnvManager


class SettingsPanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, width=450)
        self.app = app  # Main Windowへの参照

        # 変数保持
        self.common_vars = {}
        self.model_vars = {}
        self.adapter_vars = {}
        self.data_vars = {}

        self.registry = Registry()  # Registry初期化

        self.current_hash = tk.StringVar(value="Calculating...")

        self.last_hash_payload = None

        # Task Selection
        self.task_mode = tk.StringVar(value="train")
        self.selected_plot_class = tk.StringVar()

        # 表示名と実際のクラス名のマッピング保持用
        self.plot_map = {}

        self.setup_ui()
        self.update_form()

    def setup_ui(self):
        ttk.Label(self, text="🚀 Experiment Settings", font=("", 14, "bold")).pack(
            pady=10
        )

        # 1. Action Area (Bottom Fixed)
        action_area = ttk.Frame(self)
        action_area.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=10)

        # Hash Display
        hash_frame = ttk.Frame(action_area)
        hash_frame.pack(fill=tk.X, pady=5)
        ttk.Label(hash_frame, text="Hash ID:").pack(side=tk.LEFT)
        ttk.Entry(hash_frame, textvariable=self.current_hash, state="readonly").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5
        )

        # Task Selection
        task_frame = ttk.LabelFrame(action_area, text="Task Type")
        task_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(
            task_frame,
            text="Train",
            variable=self.task_mode,
            value="train",
            command=self.update_action_state,
        ).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(
            task_frame,
            text="Plot",
            variable=self.task_mode,
            value="plot",
            command=self.update_action_state,
        ).pack(side=tk.LEFT, padx=10)

        self.plot_combo = ttk.Combobox(
            task_frame, textvariable=self.selected_plot_class, state="disabled"
        )
        self.plot_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.btn_submit = ttk.Button(
            action_area, text="➕ ADD JOB", command=self.submit_job
        )
        self.btn_submit.pack(fill=tk.X, pady=5)

        # 2. Scrollable Settings Area
        self.frame_params_container = ttk.Frame(self)
        self.frame_params_container.pack(
            side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        self.scrollbar = ttk.Scrollbar(self.frame_params_container, orient="vertical")
        self.canvas = tk.Canvas(
            self.frame_params_container, yscrollcommand=self.scrollbar.set
        )
        self.scrollbar.config(command=self.canvas.yview)

        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.scroll_frame, anchor="nw"
        )

        self.scroll_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # --- Contents ---
        self.frame_static = ttk.Frame(self.scroll_frame)
        self.frame_static.pack(fill=tk.X, expand=True)

        self.frame_dynamic = ttk.Frame(self.scroll_frame)
        self.frame_dynamic.pack(fill=tk.X, expand=True)

        # Common Settings
        common_frame = ttk.LabelFrame(self.frame_static, text="Common Settings")
        common_frame.pack(fill=tk.X, pady=5, padx=2)
        self.common_schema = self.load_schema("common", "")
        self.create_fields(
            common_frame, "common", "", self.common_schema, self.common_vars
        )

        # Selection
        sel_frame = ttk.LabelFrame(self.frame_static, text="Target")
        sel_frame.pack(fill=tk.X, pady=5, padx=2)

        # Registryからリストを取得
        self.models = list(self.registry.data.get("models", {}).keys())
        self.datasets = list(self.registry.data.get("datasets", {}).keys())

        ttk.Label(sel_frame, text="Model:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.cb_model = ttk.Combobox(sel_frame, values=self.models, state="readonly")
        self.cb_model.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(sel_frame, text="Adapter:").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        self.cb_adapter = ttk.Combobox(sel_frame, state="readonly")
        self.cb_adapter.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(sel_frame, text="Dataset:").grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        self.cb_dataset = ttk.Combobox(
            sel_frame, values=self.datasets, state="readonly"
        )
        self.cb_dataset.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.cb_model.bind("<<ComboboxSelected>>", self.on_model_selected)
        self.cb_adapter.bind("<<ComboboxSelected>>", self.on_adapter_selected)
        self.cb_dataset.bind("<<ComboboxSelected>>", self.update_form)

        if self.models:
            self.cb_model.current(0)
            self.on_model_selected(None)

    # --- Event Handlers & Helpers ---
    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    # scan_definitions, scan_adapters は廃止

    def scan_adapters(self, model_name):
        # RegistryからAdapterリストを取得
        try:
            info = self.registry.get_model_info(model_name)
            return list(info.get("adapters", {}).keys())
        except:
            return []

    def load_schema(self, kind, name, sub_kind=None, sub_name=None):
        target_cls = None
        json_dir = ""

        try:
            if kind == "common":
                mod = EnvManager().get_common_config_module()
                from MLsystem.inspector import find_config_class

                target_cls = find_config_class(mod)
                json_dir = "common"
            else:
                # Registryを使用
                if sub_name:  # Adapter
                    target_cls = self.registry.get_config_class(kind, name, sub_name)
                    info = self.registry.get_adapter_info(name, sub_name)
                    json_dir = info["base_dir"]
                else:  # Model / Dataset
                    target_cls = self.registry.get_config_class(kind, name)
                    if kind == "models":
                        info = self.registry.get_model_info(name)
                    else:
                        info = self.registry.get_dataset_info(name)
                    json_dir = info["base_dir"]

            schema = {}
            if target_cls:
                # クラスオブジェクト自体を保存（ハッシュ計算用）
                schema["__class_obj__"] = target_cls

                for f in fields(target_cls):
                    schema[f.name] = {
                        "type": f.type,
                        "default": f.default,
                        "desc": f.metadata.get("desc", ""),
                        "ui_mode": f.metadata.get("ui_mode", "input"),
                        "ignore": f.metadata.get("ignore", False),
                    }
        except Exception as e:
            print(f"Error loading schema ({kind}/{name}): {e}")
            schema = {}

        # ユーザー設定(user_config.json)のロードと適用
        try:
            if json_dir:
                if not os.path.isabs(json_dir):
                    # プロジェクトルート基準（Registryの場所などから推定）
                    base_dir = os.path.dirname(EnvManager().registry_path)
                    json_dir = os.path.join(base_dir, json_dir)

                json_path = os.path.join(json_dir, "user_config.json")
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        user_vals = json.load(f)
                    for k, v in user_vals.items():
                        if k in schema:
                            schema[k]["default"] = v
        except Exception as e:
            print(f"Error loading user config: {e}")
            pass

        return schema

        #     if target_cls:
        #         # 後でクラスオブジェクト自体が必要になるため、隠しフィールドとして保持しておく
        #         schema["__class_obj__"] = target_cls

        #         for f in fields(target_cls):
        #             schema[f.name] = {
        #                 "type": f.type,
        #                 "default": f.default,
        #                 "desc": f.metadata.get("desc", ""),
        #                 "ui_mode": f.metadata.get("ui_mode", "input"),
        #                 "ignore": f.metadata.get("ignore", False),
        #             }
        # except Exception:
        #     schema = {}

        # try:
        #     json_path = os.path.join(json_dir, "user_config.json")
        #     if os.path.exists(json_path):
        #         with open(json_path, "r") as f:
        #             user_vals = json.load(f)
        #         for k, v in user_vals.items():
        #             if k in schema:
        #                 schema[k]["default"] = v
        # except: pass
        # return schema

    def create_fields(
        self, parent, kind, name, schema, var_dict, sub_kind=None, sub_name=None
    ):
        row = 0
        for key, info in schema.items():
            # 内部保持用のクラスオブジェクトはスキップ
            if key == "__class_obj__":
                continue

            if key == "_name":
                info["ui_mode"] = "hidden"
            val_type = info["type"]
            default_val = info["default"]
            ui_mode = info.get("ui_mode", "input")

            if ui_mode == "hidden":
                var = (
                    tk.BooleanVar(value=default_val)
                    if val_type is bool
                    else tk.StringVar(value=str(default_val))
                )
                var_dict[key] = (var, val_type)
                continue

            ttk.Label(parent, text=key).grid(row=row, column=0, sticky="w", padx=5)

            if val_type is bool:
                var = tk.BooleanVar(value=default_val)
                state = "disabled" if ui_mode == "readonly" else "normal"
                chk = ttk.Checkbutton(
                    parent, variable=var, command=self.recalc_hash, state=state
                )
                chk.grid(row=row, column=1, sticky="w")
                var_dict[key] = (var, val_type)
            else:
                var = tk.StringVar(value=str(default_val))
                state = "readonly" if ui_mode == "readonly" else "normal"
                entry = ttk.Entry(parent, textvariable=var, state=state)
                entry.grid(row=row, column=1, sticky="ew")
                if ui_mode != "readonly":
                    entry.bind("<KeyRelease>", lambda e: self.recalc_hash())
                var_dict[key] = (var, val_type)
            row += 1

    def on_model_selected(self, event):
        model = self.cb_model.get()
        adapters = self.scan_adapters(model)
        self.cb_adapter["values"] = adapters
        if adapters:
            self.cb_adapter.current(0)
            self.on_adapter_selected(None)
        else:
            self.cb_adapter.set("")
            self.update_form()

    def on_adapter_selected(self, event):
        model = self.cb_model.get()
        adapter = self.cb_adapter.get()
        if not model or not adapter:
            return

        schema = self.load_schema("models", model, "adapters", adapter)
        target_dataset = schema.get("target_dataset", {}).get("default")

        if target_dataset and target_dataset in self.datasets:
            self.cb_dataset.set(target_dataset)
            self.cb_dataset.configure(state="disabled")
        else:
            self.cb_dataset.configure(state="readonly")
        self.update_form()

    def update_form(self, event=None):
        for widget in self.frame_dynamic.winfo_children():
            widget.destroy()

        self.model_vars = {}
        self.adapter_vars = {}
        self.data_vars = {}

        m_name = self.cb_model.get()
        a_name = self.cb_adapter.get()
        d_name = self.cb_dataset.get()

        if not m_name:
            return

        lf_m = ttk.LabelFrame(self.frame_dynamic, text=f"Model: {m_name}")
        lf_m.pack(fill=tk.X, padx=5, pady=5)
        m_schema = self.load_schema("models", m_name)
        self.create_fields(lf_m, "models", m_name, m_schema, self.model_vars)

        if a_name:
            lf_a = ttk.LabelFrame(self.frame_dynamic, text=f"Adapter: {a_name}")
            lf_a.pack(fill=tk.X, padx=5, pady=5)
            a_schema = self.load_schema("models", m_name, "adapters", a_name)
            self.create_fields(
                lf_a, "models", m_name, a_schema, self.adapter_vars, "adapters", a_name
            )

        if d_name:
            lf_d = ttk.LabelFrame(self.frame_dynamic, text=f"Dataset: {d_name}")
            lf_d.pack(fill=tk.X, padx=5, pady=5)
            d_schema = self.load_schema("datasets", d_name)
            self.create_fields(lf_d, "datasets", d_name, d_schema, self.data_vars)

        self.recalc_hash()
        self.update_plot_list()

    def update_plot_list(self):
        m = self.cb_model.get()
        a = self.cb_adapter.get()
        d = self.cb_dataset.get()

        # リセット
        self.plot_combo.set("")
        self.plot_map = {}

        if not m:
            self.plot_combo["values"] = []
            return

        try:
            # 戻り値: [{"class": PlotClass, "target": ..., "label": ...}, ...]
            plots = get_available_plots(m, a, d)
            display_values = []

            for item in plots:
                label = item["label"]
                cls_name = item["class"].__name__
                target = item["target"]

                # マッピング保存 (表示名 -> {class: クラス名, target: ターゲットメンバ名})
                self.plot_map[label] = {"class": cls_name, "target": target}
                display_values.append(label)

            self.plot_combo["values"] = display_values

            if display_values:
                self.plot_combo.current(0)
            else:
                self.selected_plot_class.set("")

        except Exception as e:
            print(f"Error updating plot list: {e}")
            self.plot_combo["values"] = []

    def update_action_state(self):
        mode = self.task_mode.get()
        if mode == "plot":
            self.plot_combo.configure(state="readonly")
            self.btn_submit.configure(text="➕ ADD PLOT JOB")
        else:
            self.plot_combo.configure(state="disabled")
            self.btn_submit.configure(text="➕ ADD TRAIN JOB")

    def get_params(self, var_dict):
        params = {}
        for k, (var, vtype) in var_dict.items():
            val = var.get()
            if vtype is int:
                params[k] = int(float(val)) if val else 0
            elif vtype is float:
                params[k] = float(val) if val else 0.0
            elif vtype is bool:
                params[k] = bool(val)
            else:
                params[k] = val
        return params

    def recalc_hash(self):
        # GUIの入力値から一時的な設定辞書を作成
        try:
            c_p = self.get_params(self.common_vars)
            m_p = self.get_params(self.model_vars)
            a_p = self.get_params(self.adapter_vars)
            d_p = self.get_params(self.data_vars)

            m_n = self.cb_model.get()
            a_n = self.cb_adapter.get()
            d_n = self.cb_dataset.get()

            # 設定辞書を構築
            config = {
                "model": m_n,
                "adapter": a_n,
                "dataset": d_n,
                "common": c_p,
                "model_params": m_p,
                "adapter_params": a_p,
                "data_params": d_p
            }
            
            # Hydra引数に変換
            hydra_args = dict_to_hydra_args(config)
            
            # submit.py のロジックを使ってハッシュ計算
            # ※注意: これは重い処理になる可能性がある
            hid = calculate_hash(hydra_args)
            
            self.current_hash.set(hid)
            # last_hash_payload は不要になったが、互換性のため保持するなら空でもよい
            # または Config保存ロジックを別途考える必要があるが、
            # submit.py 経由で行う場合、config_diff.json の生成責務も submit/runner 側に移る。
            # GUI側での config_diff.json 保存ロジック (submit_job内) も見直す必要があるが、
            # とりあえずハッシュ表示はこれで完了。
            self.last_config_dict = config # 後の保存用にとっておく

        except Exception as e:
            print(f"Hash calculation error: {e}")
            self.current_hash.set("Error")

    def save_user_config(self, kind, name, var_dict, sub_kind=None, sub_name=None):
        if kind == "common":
            json_dir = "common"
        elif sub_kind:
            json_dir = os.path.join("definitions", kind, name, sub_kind, sub_name)
        else:
            json_dir = os.path.join("definitions", kind, name)

        # 相対パスの場合はプロジェクトルート（Registry基準）を付加
        if not os.path.isabs(json_dir):
            base_dir = os.path.dirname(EnvManager().registry_path)
            json_dir = os.path.join(base_dir, json_dir)

        os.makedirs(json_dir, exist_ok=True)
        json_path = os.path.join(json_dir, "user_config.json")
        try:
            with open(json_path, "w") as f:
                json.dump(self.get_params(var_dict), f, indent=4)
        except:
            pass

    def submit_job(self):
        hid = self.current_hash.get()
        if hid == "Error":
            return

        mode = self.task_mode.get()
        m_n = self.cb_model.get()
        a_n = self.cb_adapter.get()
        d_n = self.cb_dataset.get()

        self.save_user_config("common", "", self.common_vars)
        self.save_user_config("models", m_n, self.model_vars)
        self.save_user_config("models", m_n, self.adapter_vars, "adapters", a_n)
        self.save_user_config("datasets", d_n, self.data_vars)
        
        # hydra_args の生成は recalc_hash と同じロジックが必要
        # ここでは dict_to_hydra_args を使う
        config = {
            "model": m_n,
            "adapter": a_n,
            "dataset": d_n,
            "common": self.get_params(self.common_vars),
            "model_params": self.get_params(self.model_vars),
            "adapter_params": self.get_params(self.adapter_vars),
            "data_params": self.get_params(self.data_vars)
        }
        hydra_args = dict_to_hydra_args(config)

        # config_diff.json の保存責務について:
        # 以前はここでGUIが保存していたが、Runner/ExecuteTrainが保存するようになったため
        # ここでの保存は不要...と言いたいが、
        # "実験フォルダを作って設定を置く" のはジョブ実行前に行いたい場合もある(プレースホルダ)。
        # しかし execute_train.py が config_diff.json を上書き保存するので、
        # ここでは保存処理を削除し、完全にバックエンドに任せるのが安全。
        # (Runnerが動くまでフォルダが作られない可能性があるが、add_jobがQueueファイルを作るのでOK)

        if mode == "train":
            # hash_id は None を渡す (submit.py で計算)
            add_job(hydra_args, task_type="train", hash_id=None)
        else:
            display_label = self.selected_plot_class.get()
            if not display_label:
                messagebox.showwarning("Warning", "Please select a plot class.")
                return

            # マッピングから情報を取得
            plot_info = self.plot_map.get(display_label)
            if not plot_info:
                messagebox.showerror("Error", "Could not resolve plot class info.")
                return
            
            # plot_info は {"class": cls_name, "target": target_member}
            real_class_name = plot_info["class"]
            target_member = plot_info["target"]

            add_job(hydra_args, task_type="plot", hash_id=None, target_class=real_class_name, target_member=target_member)

        # アプリ全体に更新を通知
        self.app.on_job_submitted()