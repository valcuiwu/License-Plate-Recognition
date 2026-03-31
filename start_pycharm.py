"""PyCharm/VS Code one-click launcher for the Tkinter UI."""

import os
import runpy

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(ROOT_DIR, "Chinese_plate")

os.chdir(APP_DIR)
runpy.run_path("UI.py", run_name="__main__")
