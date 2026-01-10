# system/gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import importlib
import json
import glob
import time
from datetime import datetime
from dataclasses import fields
import inspect

# プロジェクトルートにパスを通す
sys.path.append(os.getcwd())

from system.hashing import compute_combined_hash
from system.submit import add_job, stop_runner, ensure_runner_running
from system.queue_manager import QueueManager
from system.inspector import get_available_plots
from system.config_base import BaseConfig

class ExperimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Experiment Cockpit (Desktop) v3.2")
        self.geometry("1200x900")
        
        # 変数保持
        self.common_vars = {}
        self.model_vars = {}
        self.adapter_vars = {}
        self.data_vars = {}
        self.current_hash = tk.StringVar(value="Calculating...")
        
        # ハッシュ計算結果の一時保存用
        self.last_hash_payload = None 
        
        # Task Selection
        self.task_mode = tk.StringVar(value="train") # "train" or "plot"
        self.selected_plot_class = tk.StringVar()
        
        self.qm = QueueManager()

        # --- メインレイアウト ---
        self.paned_main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned_main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左パネル: 設定 (固定幅)
        self.frame_left = ttk.Frame(self.paned_main, width=450)
        self.paned_main.add(self.frame_left, weight=1)

        # 右パネル: モニター (リサイズ可)
        self.frame_right = ttk.Frame(self.paned_main)
        self.paned_main.add(self.frame_right, weight=3)

        # 右パネルをさらに上下に分割 (Queue / History)
        self.paned_right = ttk.PanedWindow(self.frame_right, orient=tk.VERTICAL)
        self.paned_right.pack(fill=tk.BOTH, expand=True)

        self.frame_queue = ttk.Labelframe(self.paned_right, text="Job Queue Monitor")
        self.frame_history = ttk.Labelframe(self.paned_right, text="Experiment History")
        
        self.paned_right.add(self.frame_queue, weight=1)
        self.paned_right.add(self.frame_history, weight=1)

        # 各エリアの構築
        self.setup_left_panel()
        self.setup_queue_panel()
        self.setup_history_panel()

        # 初期化
        self.update_form()
        self.start_auto_refresh()

    # ==========================================================
    #  Left Panel: Experiment Settings
    # ==========================================================
    def setup_left_panel(self):
        ttk.Label(self.frame_left, text="🚀 Experiment Settings", font=("", 14, "bold")).pack(pady=10)
        
        # 1. Fixed Settings Area (Top)
        fixed_frame = ttk.Frame(self.frame_left)
        fixed_frame.pack(side=tk.TOP, fill=tk.X, padx=5)

        # Common Settings
        common_frame = ttk.LabelFrame(fixed_frame, text="Common Settings")
        common_frame.pack(fill=tk.X, pady=5)
        self.common_schema = self.load_schema("common", "")
        self.create_fields(common_frame, "common", "", self.common_schema, self.common_vars)

        # Selection
        sel_frame = ttk.LabelFrame(fixed_frame, text="Target")
        sel_frame.pack(fill=tk.X, pady=5)
        
        self.models = self.scan_definitions("models")
        self.datasets = self.scan_definitions("datasets")
        
        # Model
        ttk.Label(sel_frame, text="Model:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.cb_model = ttk.Combobox(sel_frame, values=self.models, state="readonly")
        self.cb_model.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Adapter
        ttk.Label(sel_frame, text="Adapter:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.cb_adapter = ttk.Combobox(sel_frame, state="readonly")
        self.cb_adapter.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # Dataset
        ttk.Label(sel_frame, text="Dataset:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.cb_dataset = ttk.Combobox(sel_frame, values=self.datasets, state="readonly")
        self.cb_dataset.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.cb_model.bind("<<ComboboxSelected>>", self.on_model_selected)
        self.cb_adapter.bind("<<ComboboxSelected>>", self.on_adapter_selected)
        self.cb_dataset.bind("<<ComboboxSelected>>", self.update_form)
        
        # 3. Action Area (Bottom Fixed) - Pack this BEFORE the middle scroll area
        # 下部に固定するため side=BOTTOM を使用
        action_area = ttk.Frame(self.frame_left)
        action_area.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=10)
        
        # Hash Display
        hash_frame = ttk.Frame(action_area)
        hash_frame.pack(fill=tk.X, pady=5)
        ttk.Label(hash_frame, text="Hash ID:").pack(side=tk.LEFT)
        ttk.Entry(hash_frame, textvariable=self.current_hash, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Task Selection (Train vs Plot)
        task_frame = ttk.LabelFrame(action_area, text="Task Type")
        task_frame.pack(fill=tk.X, pady=5)
        
        rb_train = ttk.Radiobutton(task_frame, text="Train", variable=self.task_mode, value="train", command=self.update_action_state)
        rb_train.pack(side=tk.LEFT, padx=10)
        
        rb_plot = ttk.Radiobutton(task_frame, text="Plot", variable=self.task_mode, value="plot", command=self.update_action_state)
        rb_plot.pack(side=tk.LEFT, padx=10)
        
        # Plot Selector (Visible/Enabled only for Plot)
        self.plot_combo = ttk.Combobox(task_frame, textvariable=self.selected_plot_class, state="disabled")
        self.plot_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Submit Button
        self.btn_submit = ttk.Button(action_area, text="➕ ADD JOB", command=self.submit_job)
        self.btn_submit.pack(fill=tk.X, pady=5)

        # 2. Scrollable Params Area (Middle, Expands)
        # スクロールバー設定用のコンテナ
        self.frame_params_container = ttk.Frame(self.frame_left)
        self.frame_params_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.scrollbar = ttk.Scrollbar(self.frame_params_container, orient="vertical")
        self.canvas = tk.Canvas(self.frame_params_container, yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.canvas.yview)
        
        # Pack順序: Scrollbarを右、Canvasを左
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.scroll_frame = ttk.Frame(self.canvas)
        # window作成時に参照を保持
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        
        # イベントバインド
        self.scroll_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        if self.models: 
            self.cb_model.current(0)
            self.on_model_selected(None)

    def on_frame_configure(self, event):
        """内部フレームのサイズが変わったらスクロール領域を更新"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """Canvasの幅が変わったら内部フレームの幅を合わせる"""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    # ==========================================================
    #  Top Right: Queue Monitor
    # ==========================================================
    def setup_queue_panel(self):
        # Runner Controls
        runner_frame = ttk.Frame(self.frame_queue)
        runner_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(runner_frame, text="▶ START RUNNER", command=self.start_runner).pack(side=tk.LEFT, padx=5)
        ttk.Button(runner_frame, text="⏹ STOP RUNNER", command=self.stop_runner).pack(side=tk.LEFT, padx=5)
        ttk.Button(runner_frame, text="🔄 Refresh", command=self.update_job_list).pack(side=tk.RIGHT, padx=5)

        # Treeview
        cols = ("Type", "ID", "Status", "Model", "Dataset", "Submitted")
        self.tree_queue = ttk.Treeview(self.frame_queue, columns=cols, show="headings", selectmode="browse")
        
        for c in cols:
            self.tree_queue.heading(c, text=c)
            width = 80 if c == "Type" or c == "Status" else 100
            if c == "ID": width = 120
            self.tree_queue.column(c, width=width)
        self.tree_queue.column("Submitted", width=120)

        ysb = ttk.Scrollbar(self.frame_queue, orient=tk.VERTICAL, command=self.tree_queue.yview)
        self.tree_queue.configure(yscroll=ysb.set)
        
        self.tree_queue.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)

        # Queue Actions
        btn_box = ttk.Frame(self.frame_queue)
        btn_box.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_box, text="⬆ Top", width=6, command=self.move_job_front).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_box, text="🗑 Delete", command=self.delete_job).pack(side=tk.LEFT, padx=2)

        # Colors
        self.tree_queue.tag_configure("running", background="#e6f7ff")
        self.tree_queue.tag_configure("pending", background="#fffbe6")
        self.tree_queue.tag_configure("failed", background="#fff1f0")
        self.tree_queue.tag_configure("plot", foreground="#000080")

    # ==========================================================
    #  Bottom Right: History
    # ==========================================================
    def setup_history_panel(self):
        # リスト表示のみ
        list_frame = ttk.Frame(self.frame_history)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        cols = ("HashID", "Model", "Adapter", "Dataset", "Status")
        self.tree_hist = ttk.Treeview(list_frame, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            self.tree_hist.heading(c, text=c)
            self.tree_hist.column(c, width=100)
            
        ysb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree_hist.yview)
        self.tree_hist.configure(yscroll=ysb.set)
        
        self.tree_hist.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        
        ttk.Button(list_frame, text="🔄 Reload Experiments", command=self.load_history).pack(side=tk.BOTTOM, fill=tk.X)

    # ==========================================================
    #  Logic: Settings & Schemas
    # ==========================================================
    def scan_definitions(self, kind):
        path = os.path.join("definitions", kind)
        if not os.path.exists(path): return []
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and not d.startswith("__")]

    def scan_adapters(self, model_name):
        path = os.path.join("definitions", "models", model_name, "adapters")
        if not os.path.exists(path): return []
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and not d.startswith("__")]

    def load_schema(self, kind, name, sub_kind=None, sub_name=None):
        if kind == "common":
            mod_path = "common.config"
            json_dir = "common"
        elif sub_kind:
            mod_path = f"definitions.{kind}.{name}.{sub_kind}.{sub_name}.config"
            json_dir = os.path.join("definitions", kind, name, sub_kind, sub_name)
        else:
            mod_path = f"definitions.{kind}.{name}.config"
            json_dir = os.path.join("definitions", kind, name)
            
        try:
            mod = importlib.import_module(mod_path)
            schema = {}

            # 1. 旧方式: CONFIG_SCHEMA があればそれを使う
            if hasattr(mod, "CONFIG_SCHEMA"):
                schema = getattr(mod, "CONFIG_SCHEMA", {}).copy()
            else:
                # 2. 新方式: Dataclass を探す (BaseConfig依存を弱めて探索)
                target_cls = None
                for name, obj in inspect.getmembers(mod, inspect.isclass):
                    # そのモジュールで定義された Dataclass を探す
                    if obj.__module__ == mod.__name__ and is_dataclass(obj):
                        target_cls = obj
                        break
                
                if target_cls:
                    for f in fields(target_cls):
                        schema[f.name] = {
                            "type": f.type,
                            "default": f.default,
                            "desc": f.metadata.get("desc", ""),
                            "ui_mode": f.metadata.get("ui_mode", "input"),
                            "ignore": f.metadata.get("ignore", False),
                        }

        except Exception as e:
            # エラーが発生した場合は原因を表示する (これが表示されない原因の可能性が高い)
            print(f"[GUI Error] Failed to load config from {mod_path}")
            print(f"Reason: {e}")
            traceback.print_exc()
            schema = {}

        try:
            # user_config.json
            json_path = os.path.join(json_dir, "user_config.json")            
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        user_vals = json.load(f)
                    for k, v in user_vals.items():
                        if k in schema:
                            schema[k]["default"] = v
                except: pass
            return schema
        except Exception:
            return {}

    def create_fields(self, parent, kind, name, schema, var_dict, sub_kind=None, sub_name=None):
        row = 0
        for key, info in schema.items():
            val_type = info["type"]
            default_val = info["default"]
            ui_mode = info.get("ui_mode", "input")

            if ui_mode == "hidden":
                var = tk.BooleanVar(value=default_val) if val_type is bool else tk.StringVar(value=str(default_val))
                var_dict[key] = (var, val_type)
                continue
            
            ttk.Label(parent, text=key).grid(row=row, column=0, sticky="w", padx=5)
            
            if val_type is bool:
                var = tk.BooleanVar(value=default_val)
                state = "disabled" if ui_mode == "readonly" else "normal"
                chk = ttk.Checkbutton(parent, variable=var, command=self.recalc_hash, state=state)
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

    def save_user_config(self, kind, name, var_dict, sub_kind=None, sub_name=None):
        if kind == "common":
            json_dir = "common"
        elif sub_kind:
            json_dir = os.path.join("definitions", kind, name, sub_kind, sub_name)
        else:
            json_dir = os.path.join("definitions", kind, name)
            
        os.makedirs(json_dir, exist_ok=True)
        json_path = os.path.join(json_dir, "user_config.json")
        current_vals = self.get_params(var_dict)
        try:
            with open(json_path, "w") as f:
                json.dump(current_vals, f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to save {json_path}: {e}")

    def on_model_selected(self, event):
        model = self.cb_model.get()
        adapters = self.scan_adapters(model)
        self.cb_adapter['values'] = adapters
        if adapters:
            self.cb_adapter.current(0)
            self.on_adapter_selected(None)
        else:
            self.cb_adapter.set("")
            self.update_form()

    def on_adapter_selected(self, event):
        model = self.cb_model.get()
        adapter = self.cb_adapter.get()
        if not model or not adapter: return
        
        schema = self.load_schema("models", model, "adapters", adapter)
        target_dataset = schema.get("target_dataset", {}).get("default")
            
        if target_dataset and target_dataset in self.datasets:
            self.cb_dataset.set(target_dataset)
            self.cb_dataset.configure(state="disabled")
        else:
            self.cb_dataset.configure(state="readonly")
            
        self.update_form()

    def update_form(self, event=None):
        # フォームの再生成
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        
        self.model_vars = {}
        self.adapter_vars = {}
        self.data_vars = {}
        
        model_name = self.cb_model.get()
        adapter_name = self.cb_adapter.get()
        data_name = self.cb_dataset.get()
        
        if not model_name: return

        # Model Params
        lf_m = ttk.LabelFrame(self.scroll_frame, text=f"Model: {model_name}")
        lf_m.pack(fill=tk.X, padx=5, pady=5)
        m_schema = self.load_schema("models", model_name)
        self.create_fields(lf_m, "models", model_name, m_schema, self.model_vars)

        # Adapter Params
        if adapter_name:
            lf_a = ttk.LabelFrame(self.scroll_frame, text=f"Adapter: {adapter_name}")
            lf_a.pack(fill=tk.X, padx=5, pady=5)
            a_schema = self.load_schema("models", model_name, "adapters", adapter_name)
            self.create_fields(lf_a, "models", model_name, a_schema, self.adapter_vars, 
                             sub_kind="adapters", sub_name=adapter_name)

        # Data Params
        if data_name:
            lf_d = ttk.LabelFrame(self.scroll_frame, text=f"Dataset: {data_name}")
            lf_d.pack(fill=tk.X, padx=5, pady=5)
            d_schema = self.load_schema("datasets", data_name)
            self.create_fields(lf_d, "datasets", data_name, d_schema, self.data_vars)
        
        self.recalc_hash()
        self.update_plot_list() # モデルが変わったのでPlotリストも更新

    def update_plot_list(self):
        """現在選択中のモデルで利用可能なPlotクラスをリストアップ"""
        model_name = self.cb_model.get()
        if not model_name:
            self.plot_combo['values'] = []
            return
            
        try:
            plots = get_available_plots(model_name)
            plot_names = [p.__name__ for p in plots]
            self.plot_combo['values'] = plot_names
            if plot_names:
                self.plot_combo.current(0)
        except Exception:
            self.plot_combo['values'] = []

    def update_action_state(self):
        """Train/Plot切り替え時のUI制御"""
        mode = self.task_mode.get()
        if mode == "plot":
            self.plot_combo.configure(state="readonly")
            self.btn_submit.configure(text="➕ ADD PLOT JOB")
        else:
            self.plot_combo.configure(state="disabled")
            self.btn_submit.configure(text="➕ ADD TRAIN JOB")

    def recalc_hash(self):
        try:
            c_params = self.get_params(self.common_vars)
            m_params = self.get_params(self.model_vars)
            a_params = self.get_params(self.adapter_vars)
            d_params = self.get_params(self.data_vars)
            
            m_name = self.cb_model.get()
            a_name = self.cb_adapter.get()
            d_name = self.cb_dataset.get()
            
            m_params["_name"] = m_name
            a_params["_name"] = a_name
            d_params["_name"] = d_name
            
            c_schema = self.common_schema
            m_schema = self.load_schema("models", m_name)
            a_schema = self.load_schema("models", m_name, "adapters", a_name)
            d_schema = self.load_schema("datasets", d_name)
            
            hid, payload = compute_combined_hash(
                c_schema, c_params,
                m_schema, m_params, 
                a_schema, a_params,
                d_schema, d_params
            )
            self.current_hash.set(hid)
            self.last_hash_payload = payload
        except Exception:
            self.current_hash.set("Error")
            self.last_hash_payload = None

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

    def submit_job(self):
        hid = self.current_hash.get()
        if hid == "Error" or not self.last_hash_payload: return
        
        mode = self.task_mode.get()
        m_name = self.cb_model.get()
        a_name = self.cb_adapter.get()
        d_name = self.cb_dataset.get()

        # 1. ユーザー設定保存
        self.save_user_config("common", "", self.common_vars)
        self.save_user_config("models", m_name, self.model_vars)
        self.save_user_config("models", m_name, self.adapter_vars, "adapters", a_name)
        self.save_user_config("datasets", d_name, self.data_vars)
        
        # 2. Hydra用引数作成
        hydra_args = [f"model={m_name}", f"adapter={a_name}", f"dataset={d_name}"]
        
        c_params = self.get_params(self.common_vars)
        m_params = self.get_params(self.model_vars)
        a_params = self.get_params(self.adapter_vars)
        d_params = self.get_params(self.data_vars)
        
        for k, v in c_params.items(): hydra_args.append(f"+common.{k}={v}")
        for k, v in m_params.items(): hydra_args.append(f"+model_params.{k}={v}")
        for k, v in a_params.items(): hydra_args.append(f"+adapter_params.{k}={v}")
        for k, v in d_params.items(): hydra_args.append(f"+data_params.{k}={v}")
        
        # 3. Configファイル (config_diff.json) の先行保存 (重要)
        # Plotジョブの場合、まだ実験ディレクトリがないとLoaderが動かないため
        exp_dir = os.path.join("output", "experiments", hid)
        os.makedirs(exp_dir, exist_ok=True)
        config_path = os.path.join(exp_dir, "config_diff.json")
        
        if not os.path.exists(config_path):
            try:
                with open(config_path, "w") as f:
                    json.dump(self.last_hash_payload, f, indent=4)
                print(f"Saved config for new experiment: {hid}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save config: {e}")
                return

        # 4. ジョブ投入
        if mode == "train":
            add_job(hydra_args, task_type="train")
        else:
            plot_cls = self.selected_plot_class.get()
            if not plot_cls:
                messagebox.showwarning("Warning", "Please select a plot class.")
                return
            
            extra = {
                "hash_id": hid,
                "target_class": plot_cls
            }
            # Plotジョブでも、不足分学習のためにhydra_argsを持たせておく
            add_job(hydra_args, task_type="plot", extra_data=extra)
            
        self.update_job_list()

    def start_runner(self):
        ensure_runner_running()
        self.update_job_list()

    def stop_runner(self):
        if stop_runner():
            messagebox.showinfo("Stopped", "Runner stopped.")
        self.update_job_list()

    # === Queue & History Monitor ===
    def start_auto_refresh(self):
        self.update_job_list()
        self.load_history()
        self.after(3000, self.start_auto_refresh)

    def update_job_list(self):
        # 選択状態の保存
        selected = self.tree_queue.selection()
        saved_sel = selected[0] if selected else None
        
        # 1. 実行中リスト
        running_files = glob.glob("queue/running/*.json")
        # 2. 待機中リスト (QueueManagerから順序通り取得)
        pending_ids = self.qm.get_list()
        
        # Treeviewクリア
        for item in self.tree_queue.get_children():
            self.tree_queue.delete(item)
            
        # Running
        for fpath in running_files:
            try:
                with open(fpath, "r") as f: data = json.load(f)
                self._insert_job_row(data, "running")
            except: pass
            
        # Pending
        for jid in pending_ids:
            fpath = os.path.join("queue", "pending", f"job_{jid}.json")
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r") as f: data = json.load(f)
                    self._insert_job_row(data, "pending")
                except: pass

        # 復元
        if saved_sel and self.tree_queue.exists(saved_sel):
            self.tree_queue.selection_set(saved_sel)

    def _insert_job_row(self, data, status):
        jid = data.get("id", "???")
        ttype = data.get("task_type", "train").upper()
        args = data.get("args", [])
        
        # モデル名などの抽出
        model = next((a.split("=")[1] for a in args if a.startswith("model=")), "-")
        dataset = next((a.split("=")[1] for a in args if a.startswith("dataset=")), "-")
        
        sub_at = data.get("submitted_at", 0)
        dt = datetime.fromtimestamp(sub_at).strftime("%m/%d %H:%M")
        
        tag = "plot" if ttype == "PLOT" else status
        self.tree_queue.insert("", "end", iid=jid, values=(ttype, jid, status.upper(), model, dataset, dt), tags=(tag, status))

    def load_history(self):
        # 実験結果一覧
        root = os.path.join("output", "experiments")
        if not os.path.exists(root): return
        
        # 現在の選択を保存
        sel = self.tree_hist.selection()
        
        for item in self.tree_hist.get_children():
            self.tree_hist.delete(item)
            
        for d in os.listdir(root):
            path = os.path.join(root, d)
            conf_path = os.path.join(path, "config_diff.json")
            if os.path.isdir(path) and os.path.exists(conf_path):
                try:
                    with open(conf_path, "r") as f: conf = json.load(f)
                    m = conf.get("model", "-")
                    a = conf.get("adapter", "-")
                    d_set = conf.get("dataset", "-")
                    
                    status = "Ready"
                    if os.path.exists(os.path.join(path, "done")):
                        status = "Finished"
                    
                    # 更新日時
                    mtime = os.path.getmtime(conf_path)
                    dt = datetime.fromtimestamp(mtime).strftime("%Y/%m/%d %H:%M")
                    
                    self.tree_hist.insert("", "end", iid=d, values=(d, m, a, d_set, status))
                except: pass
        
        if sel and self.tree_hist.exists(sel[0]):
            self.tree_hist.selection_set(sel[0])

    def on_history_select(self, event):
        # 履歴選択時に何か情報を表示したければここに記述
        pass

    def move_job_front(self):
        sel = self.tree_queue.selection()
        if not sel: return
        jid = sel[0]
        # QueueManager経由で移動
        self.qm.insert_front(jid)
        self.update_job_list()

    def delete_job(self):
        sel = self.tree_queue.selection()
        if not sel: return
        jid = sel[0]
        
        # RunningかPendingか判定
        tags = self.tree_queue.item(jid, "tags")
        if "running" in tags:
            messagebox.showwarning("Warning", "Cannot delete running job. Stop runner first.")
            return
            
        # 削除
        self.qm.remove(jid)
        fpath = os.path.join("queue", "pending", f"job_{jid}.json")
        if os.path.exists(fpath):
            os.remove(fpath)
        self.update_job_list()

if __name__ == "__main__":
    app = ExperimentApp()
    app.mainloop()