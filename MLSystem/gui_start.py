import os
import sys
import tkinter as tk

# 環境変数を設定し、sys.pathに必要なパスを追加


from MLSystem.gui.main_window import ExperimentApp

if __name__ == "__main__":
  app = ExperimentApp()
  app.mainloop()