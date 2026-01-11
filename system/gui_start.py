import os
import sys
import tkinter as tk

# プロジェクトルートにパスを通す
sys.path.append(os.getcwd())

from system.gui.main_window import ExperimentApp

if __name__ == "__main__":
    app = ExperimentApp()
    app.mainloop()