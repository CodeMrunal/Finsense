"""
Run Streamlit dashboard.
"""
from curses.ascii import FF
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])













