import tkinter as tk
from tkinter import ttk
import os
import json
from datetime import datetime

class HistoryPanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.setup_ui()
        self.start_auto_refresh()

    def setup_ui(self):
        # 1. 上部アクションエリア (Monitorと同様の配置)
        action_frame = ttk.Frame(self)
        action_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # ボタンを右寄せで配置
        ttk.Button(action_frame, text="🔄 Reload Experiments", command=self.load_history).pack(side=tk.RIGHT, padx=5)

        # 2. リスト表示エリア
        list_frame = ttk.Frame(self)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        cols = ("HashID", "Model", "Adapter", "Dataset", "Status")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=100)
            
        ysb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=ysb.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)

    def start_auto_refresh(self):
        self.load_history()
        self.after(5000, self.start_auto_refresh)

    def load_history(self):
        root = os.path.join("output", "experiments")
        if not os.path.exists(root): return
        
        sel = self.tree.selection()
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for d in os.listdir(root):
            path = os.path.join(root, d)
            conf_path = os.path.join(path, "config_diff.json")
            if os.path.isdir(path) and os.path.exists(conf_path):
                try:
                    with open(conf_path, "r") as f: conf = json.load(f)
                    m = conf.get("model", "-")
                    a = conf.get("adapter", "-")
                    d_set = conf.get("dataset", "-")
                    status = "Finished" if os.path.exists(os.path.join(path, "done")) else "Ready"
                    self.tree.insert("", "end", iid=d, values=(d, m, a, d_set, status))
                except: pass
        
        if sel and self.tree.exists(sel[0]):
            self.tree.selection_set(sel[0])