import os
import sys
import tkinter as tk

# 環境変数を設定し、sys.pathに必要なパスを追加
import env_setup
env_setup.add_to_sys_path()

from gui.main_window import ExperimentApp

if __name__ == "__main__":
  app = ExperimentApp()
  app.mainloop()