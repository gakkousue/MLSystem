import tkinter as tk
from tkinter import ttk
import sys
import os

# 環境変数を設定し、sys.pathに必要なパスを追加
import env_setup
env_setup.add_to_sys_path()

from gui.panels.settings_panel import SettingsPanel
from gui.panels.queue_panel import QueuePanel
from gui.panels.history_panel import HistoryPanel

class ExperimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Experiment Cockpit (Desktop) v3.3")
        self.geometry("1280x720")
        
        # --- メインレイアウト ---
        self.paned_main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned_main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 1. 左側: 設定パネル
        self.settings_panel = SettingsPanel(self.paned_main, self)
        self.paned_main.add(self.settings_panel, weight=1)

        # 2. 右側: タブエリア (Notebook)
        self.notebook_right = ttk.Notebook(self.paned_main)
        self.paned_main.add(self.notebook_right, weight=3)
        
        # 2-A. Queue Tab
        self.queue_panel = QueuePanel(self.notebook_right, self)
        self.notebook_right.add(self.queue_panel, text="Job Queue Monitor")
        
        # 2-B. History Tab
        self.history_panel = HistoryPanel(self.notebook_right, self)
        self.notebook_right.add(self.history_panel, text="Experiment History")
        
    def on_job_submitted(self):
        """Job投入時に呼び出され、Queueリストを即時更新し、タブを切り替える"""
        self.queue_panel.update_list()
        self.notebook_right.select(self.queue_panel)