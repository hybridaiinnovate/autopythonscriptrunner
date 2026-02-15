#!/usr/bin/env python3
import os
import re
import sys
import subprocess
import threading
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QTextEdit, QPushButton, QFileDialog, QMessageBox
)


APP_NAME = "PyRunner"
MAX_RETRIES = 10


def venv_python(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def create_venv(venv_path: Path):
    if venv_path.exists():
        return
    import venv
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(str(venv_path))


def parse_missing_module(output):
    match = re.search(
        r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
        output
    )
    return match.group(1) if match else None


class Signals(QObject):
    log = Signal(str)
    status = Signal(str)
    finished = Signal(int)


class PyRunner(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(800, 600)
        self.setAcceptDrops(True)

        self.signals = Signals()
        self.signals.log.connect(self.append_log)
        self.signals.status.connect(self.set_status)
        self.signals.finished.connect(self.on_finished)

        self.thread = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self.label = QLabel("Drag & Drop a Python (.py) file or folder here")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(
            "QLabel { border: 2px dashed gray; padding: 30px; font-size: 16px; }"
        )
        layout.addWidget(self.label)

        self.pick_btn = QPushButton("Select Python File")
        self.pick_btn.clicked.connect(self.pick_file)
        layout.addWidget(self.pick_btn)

        self.pick_folder_btn = QPushButton("Select Folder")
        self.pick_folder_btn.clicked.connect(self.pick_folder)
        layout.addWidget(self.pick_folder_btn)

        self.status_label = QLabel("Status: Idle")
        layout.addWidget(self.status_label)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

    # Drag & Drop
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        paths = [Path(url.toLocalFile()) for url in urls]
        self.process_paths(paths)

    def pick_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Python File", "", "Python Files (*.py)"
        )
        if file:
            self.process_paths([Path(file)])

    def pick_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder", ""
        )
        if folder:
            self.process_paths([Path(folder)])

    def append_log(self, text):
        self.output.append(text)

    def set_status(self, text):
        self.status_label.setText(f"Status: {text}")

    def on_finished(self, code):
        self.set_status(f"Finished (exit code {code})")

    def process_paths(self, paths):
        """Process dropped files and folders - find all Python files and run them."""
        python_files = []
        
        for path in paths:
            if path.is_file() and path.suffix == ".py":
                python_files.append(path)
            elif path.is_dir():
                # Find all .py files in the folder
                python_files.extend(path.rglob("*.py"))
        
        if not python_files:
            QMessageBox.warning(self, APP_NAME, "No Python (.py) files found.")
            return
        
        self.output.clear()
        self.set_status(f"Found {len(python_files)} Python file(s)")
        
        self.thread = threading.Thread(
            target=self.workflow_multiple,
            args=(python_files,),
            daemon=True
        )
        self.thread.start()

    def run_script(self, script_path: Path):
        if not script_path.exists() or script_path.suffix != ".py":
            QMessageBox.warning(self, APP_NAME, "Please drop a valid .py file.")
            return

        self.output.clear()
        self.set_status("Preparing environment...")

        self.thread = threading.Thread(
            target=self.workflow,
            args=(script_path,),
            daemon=True
        )
        self.thread.start()

    def workflow(self, script_path: Path):
        project_dir = script_path.parent
        venv_path = project_dir / ".pyrunner_venv"

        self.signals.log.emit("Creating virtual environment...")
        create_venv(venv_path)

        py_exec = venv_python(venv_path)

        retries = 0
        while retries < MAX_RETRIES:

            self.signals.status.emit("Running script...")
            process = subprocess.Popen(
                [str(py_exec), str(script_path)],
                cwd=str(project_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            output = ""
            for line in process.stdout:
                output += line
                self.signals.log.emit(line.rstrip())

            process.wait()

            if process.returncode == 0:
                self.signals.finished.emit(0)
                return

            missing = parse_missing_module(output)
            if not missing:
                self.signals.finished.emit(process.returncode)
                return

            self.signals.log.emit(f"Installing missing module: {missing}")
            subprocess.run(
                [str(py_exec), "-m", "pip", "install", missing],
                cwd=str(project_dir)
            )
            retries += 1

        self.signals.finished.emit(1)

    def workflow_multiple(self, python_files: list):
        """Run multiple Python files sequentially."""
        # Use the directory of the first file as the project directory
        if not python_files:
            return
        
        project_dir = python_files[0].parent
        venv_path = project_dir / ".pyrunner_venv"
        
        self.signals.log.emit(f"Found {len(python_files)} Python file(s) in {project_dir}")
        self.signals.log.emit("Creating virtual environment...")
        create_venv(venv_path)
        
        py_exec = venv_python(venv_path)
        
        total = len(python_files)
        for idx, script_path in enumerate(python_files, 1):
            self.signals.log.emit(f"\n--- Running {script_path.name} ({idx}/{total}) ---")
            self.signals.status.emit(f"Running {script_path.name} ({idx}/{total})...")
            
            retries = 0
            while retries < MAX_RETRIES:
                process = subprocess.Popen(
                    [str(py_exec), str(script_path)],
                    cwd=str(project_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                output = ""
                if process.stdout:
                    for line in process.stdout:
                        output += line
                        self.signals.log.emit(line.rstrip())
                
                process.wait()
                
                if process.returncode == 0:
                    break
                
                missing = parse_missing_module(output)
                if not missing:
                    self.signals.log.emit(f"Error: Script exited with code {process.returncode}")
                    break
                
                self.signals.log.emit(f"Installing missing module: {missing}")
                subprocess.run(
                    [str(py_exec), "-m", "pip", "install", missing],
                    cwd=str(project_dir)
                )
                retries += 1
        
        self.signals.finished.emit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PyRunner()
    window.show()
    sys.exit(app.exec())
