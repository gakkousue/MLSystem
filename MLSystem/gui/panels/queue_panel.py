import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import sys
import glob
from datetime import datetime

# 環境変数を設定し、sys.pathに必要なパスを追加


from MLsystem.queue_manager import QueueManager
from MLsystem.submit import ensure_runner_running, stop_runner
from MLsystem.utils.env_manager import EnvManager


class QueuePanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.qm = QueueManager()
        self.setup_ui()
        self.start_auto_refresh()

    def setup_ui(self):
        # 1. 上部: Controls (Runner & Job Actions)
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Runner Controls
        ttk.Label(control_frame, text="Runner:").pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(control_frame, text="▶ Start", command=self.start_runner).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(control_frame, text="⏹ Stop", command=self.stop_runner).pack(
            side=tk.LEFT, padx=2
        )

        # Spacing
        ttk.Frame(control_frame, width=20).pack(side=tk.LEFT)

        # Job Actions
        ttk.Label(control_frame, text="Job:").pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(
            control_frame, text="⬆", width="auto", command=self.move_job_up
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            control_frame, text="⬇", width="auto", command=self.move_job_down
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="🗑 Delete", command=self.delete_job).pack(
            side=tk.LEFT, padx=5
        )

        # Refresh (Right aligned)
        ttk.Button(control_frame, text="🔄 Refresh", command=self.update_list).pack(
            side=tk.RIGHT, padx=5
        )

        # 2. 中央: Treeview & Scrollbar
        list_frame = ttk.Frame(self)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=0)

        cols = ("Type", "ID", "Status", "Model", "Dataset", "Submitted")
        self.tree = ttk.Treeview(
            list_frame, columns=cols, show="headings", selectmode="browse"
        )

        for c in cols:
            self.tree.heading(c, text=c)
            width = 80 if c == "Type" or c == "Status" else 100
            if c == "ID":
                width = 120
            self.tree.column(c, width=width)
        self.tree.column("Submitted", width=120)

        ysb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=ysb.set)

        # フレーム内でリストを左、スクロールバーを右に配置
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.tag_configure("running", background="#e6f7ff")
        self.tree.tag_configure("pending", background="#fffbe6")
        self.tree.tag_configure("failed", background="#fff1f0")
        self.tree.tag_configure("plot", foreground="#000080")

    def start_auto_refresh(self):
        self.update_list()
        self.after(3000, self.start_auto_refresh)

    def start_runner(self):
        ensure_runner_running()
        self.update_list()

    def stop_runner(self):
        if stop_runner():
            messagebox.showinfo("Stopped", "Runner stopped.")
        self.update_list()

    def update_list(self):
        selected = self.tree.selection()
        saved_sel = selected[0] if selected else None

        running_files = glob.glob(
            os.path.join(EnvManager().queue_dir, "running", "*.json")
        )
        pending_ids = self.qm.get_list()

        for item in self.tree.get_children():
            self.tree.delete(item)

        for fpath in running_files:
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                self._insert_row(data, "running")
            except:
                pass

        for jid in pending_ids:
            fpath = os.path.join(EnvManager().queue_dir, "pending", f"job_{jid}.json")
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r") as f:
                        data = json.load(f)
                    self._insert_row(data, "pending")
                except:
                    pass

        if saved_sel and self.tree.exists(saved_sel):
            self.tree.selection_set(saved_sel)

    def _insert_row(self, data, status):
        jid = data.get("id", "???")
        ttype = data.get("task_type", "train").upper()
        args = data.get("args", [])

        model = next((a.split("=")[1] for a in args if a.startswith("model=")), "-")
        dataset = next((a.split("=")[1] for a in args if a.startswith("dataset=")), "-")

        sub_at = data.get("submitted_at", 0)
        dt = datetime.fromtimestamp(sub_at).strftime("%m/%d %H:%M")

        tag = "plot" if ttype == "PLOT" else status
        self.tree.insert(
            "",
            "end",
            iid=jid,
            values=(ttype, jid, status.upper(), model, dataset, dt),
            tags=(tag, status),
        )

    def move_job_up(self):
        sel = self.tree.selection()
        if not sel:
            return
        self.qm.move_up(sel[0])
        self.update_list()

    def move_job_down(self):
        sel = self.tree.selection()
        if not sel:
            return
        self.qm.move_down(sel[0])
        self.update_list()

    def delete_job(self):
        sel = self.tree.selection()
        if not sel:
            return
        jid = sel[0]
        tags = self.tree.item(jid, "tags")
        if "running" in tags:
            messagebox.showwarning(
                "Warning", "Cannot delete running job. Stop runner first."
            )
            return
        self.qm.remove(jid)
        fpath = os.path.join(EnvManager().queue_dir, "pending", f"job_{jid}.json")
        if os.path.exists(fpath):
            os.remove(fpath)
        self.update_list()
