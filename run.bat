@echo off
if exist py_runner.exe (
    py_runner.exe main.py %*
) else if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe main.py %*
) else (
    python main.py %*
)
