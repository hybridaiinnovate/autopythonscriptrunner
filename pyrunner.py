#!/usr/bin/env python3
import os
import sys
import re
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from PySide6.QtCore import Qt, Signal, QObject, QEvent
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QInputDialog,
)

APP_NAME = "PyRunner"
MAX_INSTALL_RETRIES = 3
VENV_DIRNAME = ".pyrunner_venv"

IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    VENV_DIRNAME,
    "node_modules",
    "dist",
    "build",
}

# ---------------- venv helpers ----------------
def venv_python(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def create_venv(venv_path: Path) -> None:
    if venv_path.exists():
        return
    import venv

    builder = venv.EnvBuilder(with_pip=True)
    builder.create(str(venv_path))


# ---------------- project detection ----------------
def find_project_root(start: Path) -> Path:
    """Walk upwards to find the nearest directory that looks like a Python project root."""
    start = start.resolve()
    if start.is_file():
        start = start.parent

    cur = start
    while True:
        if (
            (cur / "pyproject.toml").exists()
            or (cur / "requirements.txt").exists()
            or (cur / "setup.py").exists()
            or (cur / "setup.cfg").exists()
        ):
            return cur
        if cur.parent == cur:
            return start
        cur = cur.parent


def list_root_entry_scripts(project_root: Path) -> List[Path]:
    """Common root scripts that might be intended as entrypoints."""
    names = ("main.py", "app.py", "run.py", "server.py", "cli.py", "manage.py", "wsgi.py", "asgi.py")
    return [project_root / n for n in names if (project_root / n).exists()]


def detect_module_entry(project_root: Path) -> Optional[List[str]]:
    """
    If exactly one top-level package has __main__.py, run `python -m pkg`.
    """
    pkgs = []
    for child in project_root.iterdir():
        if not child.is_dir():
            continue
        if child.name in IGNORED_DIRS:
            continue
        if (child / "__init__.py").exists() and (child / "__main__.py").exists():
            pkgs.append(child.name)
    if len(pkgs) == 1:
        return ["-m", pkgs[0]]
    return None


# ---------------- lightweight file inspection ----------------
def _safe_read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _is_fastapi_file(py_file: Path) -> bool:
    src = _safe_read_text(py_file)
    if "fastapi" not in src.lower():
        return False
    return ("FastAPI(" in src) and ("from fastapi import FastAPI" in src or "import fastapi" in src)


def _is_flask_file(py_file: Path) -> bool:
    src = _safe_read_text(py_file)
    if "flask" not in src.lower():
        return False
    return ("Flask(" in src) and ("from flask import Flask" in src or "import flask" in src)


def _find_fastapi_symbols(py_file: Path) -> List[str]:
    """
    Look for assignments like `app = FastAPI(...)`. If the file looks like FastAPI but no obvious
    assignment is found, default to `app`.
    """
    src = _safe_read_text(py_file)
    syms = []
    for name in ("app", "api", "application"):
        if re.search(rf"^\s*{name}\s*=\s*FastAPI\s*\(", src, flags=re.MULTILINE):
            syms.append(name)
    if not syms and _is_fastapi_file(py_file):
        syms.append("app")
    return syms


def _find_flask_symbols(py_file: Path) -> List[str]:
    """
    Look for assignments like `app = Flask(...)`. If the file looks like Flask but no obvious
    assignment is found, default to `app`.
    """
    src = _safe_read_text(py_file)
    syms = []
    for name in ("app", "application"):
        if re.search(rf"^\s*{name}\s*=\s*Flask\s*\(", src, flags=re.MULTILINE):
            syms.append(name)
    if not syms and _is_flask_file(py_file):
        syms.append("app")
    return syms


def _module_path_from_file(project_root: Path, py_file: Path) -> Optional[str]:
    """
    Convert a file path into a dotted module path.

    Handles common layouts:
    - project_root/pkg/mod.py -> pkg.mod
    - project_root/src/pkg/mod.py -> pkg.mod  (src layout)
    """
    project_root = project_root.resolve()
    py_file = py_file.resolve()

    # Prefer src layout if present and file is under it
    src_root = project_root / "src"
    if src_root.exists():
        try:
            rel = py_file.relative_to(src_root)
            parts = list(rel.parts)
        except Exception:
            parts = None
    else:
        parts = None

    if parts is None:
        try:
            rel = py_file.relative_to(project_root)
            parts = list(rel.parts)
        except Exception:
            return None

    if not parts or not parts[-1].endswith(".py"):
        return None

    parts[-1] = parts[-1][:-3]  # strip .py
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    parts = [p for p in parts if p and p not in IGNORED_DIRS]
    if not parts:
        return None
    return ".".join(parts)


# ---------------- CLI scripts (pyproject/setup.cfg) ----------------
def _toml_load(path: Path) -> Dict:
    txt = _safe_read_text(path)
    if not txt.strip():
        return {}
    try:
        import tomllib  # py3.11+
        return tomllib.loads(txt)
    except Exception:
        try:
            import tomli
            return tomli.loads(txt)
        except Exception:
            return {}


def _cli_scripts_from_pyproject(project_root: Path) -> List[Tuple[str, str]]:
    pyproject = project_root / "pyproject.toml"
    if not pyproject.exists():
        return []
    obj = _toml_load(pyproject)
    scripts = (obj.get("project") or {}).get("scripts") or {}
    out: List[Tuple[str, str]] = []
    if isinstance(scripts, dict):
        for k, v in scripts.items():
            if isinstance(v, str) and ":" in v:
                out.append((k, v))
    return out


def _cli_scripts_from_setup_cfg(project_root: Path) -> List[Tuple[str, str]]:
    setup_cfg = project_root / "setup.cfg"
    if not setup_cfg.exists():
        return []
    try:
        from configparser import ConfigParser
        cp = ConfigParser()
        cp.read(setup_cfg, encoding="utf-8")
    except Exception:
        return []

    if not cp.has_section("options.entry_points"):
        return []

    raw = cp.get("options.entry_points", "console_scripts", fallback="").strip()
    out: List[Tuple[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            name, target = [x.strip() for x in line.split("=", 1)]
            if ":" in target:
                out.append((name, target))
    return out


# ---------------- main entrypoint detection ----------------
def detect_entrypoints_all(project_root: Path) -> List[Tuple[str, List[str]]]:
    """
    Return list of (label, argv_without_python) with detection for:
    - FastAPI/Uvicorn
    - Django
    - Flask
    - CLI tools
    - Module packages
    - General scripts
    """
    cands: List[Tuple[str, List[str]]] = []

    # 1) Django
    manage_py = project_root / "manage.py"
    if manage_py.exists():
        cands.append(("Django: python manage.py runserver", ["manage.py", "runserver"]))
        cands.append(("Django: python manage.py check", ["manage.py", "check"]))
        cands.append(("Django: python manage.py test", ["manage.py", "test"]))

    # 2) CLI tools (pyproject / setup.cfg)
    for name, target in _cli_scripts_from_pyproject(project_root):
        mod, func = target.split(":", 1)
        # Call the entry function without arguments; many CLIs accept args, but this is a sane default.
        cands.append(
            (f"CLI: {name} (from pyproject) -> {target}",
             ["-c", f"from {mod} import {func} as _f; _f()"])
        )

    for name, target in _cli_scripts_from_setup_cfg(project_root):
        mod, func = target.split(":", 1)
        cands.append(
            (f"CLI: {name} (from setup.cfg) -> {target}",
             ["-c", f"from {mod} import {func} as _f; _f()"])
        )

    # 3) Module entry (package/__main__.py)
    mod_entry = detect_module_entry(project_root)
    if mod_entry:
        cands.append((f"Module: python {' '.join(mod_entry)}", mod_entry))

    # 4) FastAPI / Uvicorn detection
    search_dirs = [project_root]
    for dname in ("src", "app", "backend", "api"):
        p = project_root / dname
        if p.exists() and p.is_dir() and p.name not in IGNORED_DIRS:
            search_dirs.append(p)

    fastapi_files: List[Path] = []
    for base in search_dirs:
        for py in base.rglob("*.py"):
            if any(part in IGNORED_DIRS for part in py.parts):
                continue
            if py.name.startswith("."):
                continue
            if _is_fastapi_file(py):
                fastapi_files.append(py)

    # limit to avoid giant repos
    for py in fastapi_files[:12]:
        module = _module_path_from_file(project_root, py)
        if not module:
            continue
        for sym in _find_fastapi_symbols(py):
            cands.append(
                (f"FastAPI: uvicorn {module}:{sym}",
                 ["-m", "uvicorn", f"{module}:{sym}", "--host", "127.0.0.1", "--port", "8000"])
            )
            cands.append(
                (f"FastAPI (reload): uvicorn {module}:{sym} --reload",
                 ["-m", "uvicorn", f"{module}:{sym}", "--reload", "--host", "127.0.0.1", "--port", "8000"])
            )

    # 5) Flask detection
    flask_files: List[Path] = []
    for base in search_dirs:
        for py in base.rglob("*.py"):
            if any(part in IGNORED_DIRS for part in py.parts):
                continue
            if py.name.startswith("."):
                continue
            if _is_flask_file(py):
                flask_files.append(py)

    for py in flask_files[:12]:
        module = _module_path_from_file(project_root, py)
        if not module:
            continue
        for sym in _find_flask_symbols(py):
            cands.append(
                (f"Flask: flask run --app {module}:{sym}",
                 ["-m", "flask", "run", "--app", f"{module}:{sym}", "--host", "127.0.0.1", "--port", "5000"])
            )

    # 6) General scripts (root)
    for p in list_root_entry_scripts(project_root):
        # avoid double-adding manage.py if already used for Django
        if p.name == "manage.py" and manage_py.exists():
            continue
        cands.append((f"Script: python {p.name}", [p.name]))

    # 7) Last resort: any root .py (excluding setup.py)
    root_pys = sorted([p for p in project_root.glob("*.py") if p.name not in {"setup.py"}])
    for p in root_pys:
        if p.name not in {"main.py", "app.py", "run.py", "server.py", "cli.py", "manage.py", "wsgi.py", "asgi.py"}:
            cands.append((f"Root script: python {p.name}", [p.name]))

    # Deduplicate by argv
    seen = set()
    uniq: List[Tuple[str, List[str]]] = []
    for label, argv in cands:
        key = tuple(argv)
        if key not in seen:
            seen.add(key)
            uniq.append((label, argv))
    return uniq


# ---------------- missing module parse ----------------
def parse_missing_module(output: str) -> Optional[str]:
    """
    Best-effort parse for ModuleNotFoundError.
    Note: module name != pip package name sometimes; this is heuristic only.
    """
    m = re.search(r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]", output)
    if not m:
        return None
    return m.group(1)


# ---------------- Qt signals ----------------
class Signals(QObject):
    log = Signal(str)
    status = Signal(str)
    finished = Signal(int)


# ---------------- UI-thread callable event (safe) ----------------
class _CallOnUiEvent(QEvent):
    TYPE = QEvent.Type(QEvent.registerEventType())

    def __init__(self, fn):
        super().__init__(self.TYPE)
        self.fn = fn


# ---------------- Main window ----------------
class PyRunner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(900, 650)
        self.setAcceptDrops(True)

        self.signals = Signals()
        self.signals.log.connect(self.append_log)
        self.signals.status.connect(self.set_status)
        self.signals.finished.connect(self.on_finished)

        self.worker_thread: Optional[threading.Thread] = None
        self.current_process: Optional[subprocess.Popen] = None
        self.stop_requested = False

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self.label = QLabel(
            "Drag & Drop a Python PROJECT FOLDER here\n"
            "or select a folder/file below"
        )
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("QLabel { border: 2px dashed gray; padding: 26px; font-size: 15px; }")
        layout.addWidget(self.label)

        self.pick_project_btn = QPushButton("Select Project Folder (recommended)")
        self.pick_project_btn.clicked.connect(self.pick_project_folder)
        layout.addWidget(self.pick_project_btn)

        self.pick_file_btn = QPushButton("Select Single Python File (.py)")
        self.pick_file_btn.clicked.connect(self.pick_file)
        layout.addWidget(self.pick_file_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_run)
        layout.addWidget(self.stop_btn)

        self.status_label = QLabel("Status: Idle")
        layout.addWidget(self.status_label)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

    # -------- UI helpers --------
    def append_log(self, text: str) -> None:
        self.output.append(text)

    def set_status(self, text: str) -> None:
        self.status_label.setText(f"Status: {text}")

    def on_finished(self, code: int) -> None:
        self.stop_btn.setEnabled(False)
        self.current_process = None
        self.stop_requested = False
        self.set_status(f"Finished (exit code {code})")

    def stop_run(self) -> None:
        self.stop_requested = True
        self.signals.status.emit("Stopping...")
        proc = self.current_process
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    # -------- Drag & Drop --------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        paths = [Path(u.toLocalFile()) for u in urls if u.toLocalFile()]
        if not paths:
            return

        folders = [p for p in paths if p.is_dir()]
        if folders:
            self.run_project(folders[0])
            return

        files = [p for p in paths if p.is_file() and p.suffix == ".py"]
        if files:
            self.run_single_file(files[0])
            return

        QMessageBox.warning(self, APP_NAME, "Drop a folder (project) or a .py file.")

    # -------- Pickers --------
    def pick_project_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Project Folder", "")
        if folder:
            self.run_project(Path(folder))

    def pick_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Python File", "", "Python Files (*.py)")
        if file:
            self.run_single_file(Path(file))

    # -------- Run entrypoints --------
    def run_project(self, folder: Path) -> None:
        folder = folder.resolve()
        if not folder.exists() or not folder.is_dir():
            QMessageBox.warning(self, APP_NAME, "Please select a valid folder.")
            return

        self.output.clear()
        self.set_status("Preparing project...")
        self.stop_btn.setEnabled(True)
        self.stop_requested = False

        self.worker_thread = threading.Thread(target=self.workflow_project, args=(folder,), daemon=True)
        self.worker_thread.start()

    def run_single_file(self, script: Path) -> None:
        script = script.resolve()
        if not script.exists() or script.suffix != ".py":
            QMessageBox.warning(self, APP_NAME, "Please select a valid .py file.")
            return

        self.output.clear()
        self.set_status("Preparing file run...")
        self.stop_btn.setEnabled(True)
        self.stop_requested = False

        self.worker_thread = threading.Thread(target=self.workflow_single_file, args=(script,), daemon=True)
        self.worker_thread.start()

    # -------- Subprocess helpers --------
    def _run_and_stream(
        self,
        cmd: List[str],
        cwd: Path,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, str]:
        """
        Run a command, stream output live to UI, and also return full output as a string.
        """
        self.signals.log.emit(f"$ {' '.join(cmd)}")
        output_all = ""

        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=merged_env,
        )
        self.current_process = proc

        if proc.stdout:
            for line in proc.stdout:
                line = line.rstrip("\n")
                output_all += line + "\n"
                self.signals.log.emit(line)

                if self.stop_requested:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    break

        proc.wait()
        return proc.returncode, output_all

    def _pip_install(self, py_exec: Path, cwd: Path, args: List[str]) -> int:
        cmd = [str(py_exec), "-m", "pip", *args]
        code, _ = self._run_and_stream(cmd, cwd=cwd)
        return code

    def _install_deps_for_project(self, py_exec: Path, project_root: Path) -> bool:
        req = project_root / "requirements.txt"
        pyproject = project_root / "pyproject.toml"
        setup_py = project_root / "setup.py"
        setup_cfg = project_root / "setup.cfg"

        self.signals.status.emit("Upgrading pip tools...")
        self._pip_install(py_exec, project_root, ["install", "--upgrade", "pip", "setuptools", "wheel"])

        if req.exists():
            self.signals.status.emit("Installing requirements.txt...")
            rc = self._pip_install(py_exec, project_root, ["install", "-r", str(req)])
            return rc == 0

        if pyproject.exists() or setup_py.exists() or setup_cfg.exists():
            self.signals.status.emit("Installing project (pip install .)...")
            rc = self._pip_install(py_exec, project_root, ["install", str(project_root)])
            return rc == 0

        self.signals.log.emit("No requirements.txt / pyproject.toml / setup.py / setup.cfg found. Skipping dependency install.")
        return True

    # -------- UI-thread choice (safe) --------
    def event(self, event) -> bool:
        if event.type() == _CallOnUiEvent.TYPE:
            try:
                event.fn()
            except Exception as e:
                self.signals.log.emit(f"UI callback error: {e}")
            return True
        return super().event(event)

    def _ask_user_choice(self, title: str, label: str, items: List[str]) -> Optional[str]:
        """
        Show QInputDialog on the UI thread, synchronously from a worker thread.
        """
        if not items:
            return None

        result = {"choice": None}
        done = threading.Event()

        def ask():
            choice, ok = QInputDialog.getItem(self, title, label, items, 0, False)
            result["choice"] = choice if ok else None
            done.set()

        QApplication.instance().postEvent(self, _CallOnUiEvent(ask))
        done.wait()
        return result["choice"]

    # -------- Workflows --------
    def workflow_project(self, folder: Path) -> None:
        try:
            project_root = find_project_root(folder)
            self.signals.log.emit(f"Project root: {project_root}")

            venv_path = project_root / VENV_DIRNAME
            self.signals.status.emit("Creating virtual environment...")
            create_venv(venv_path)
            py_exec = venv_python(venv_path)

            ok = self._install_deps_for_project(py_exec, project_root)
            if not ok:
                self.signals.log.emit("Dependency installation failed. Aborting.")
                self.signals.finished.emit(1)
                return

            cands = detect_entrypoints_all(project_root)
            if not cands:
                self.signals.log.emit("No entrypoint detected.")
                self.signals.log.emit(
                    "Try adding one of: manage.py (Django), main.py/app.py/run.py/server.py, "
                    "a package with __main__.py, or FastAPI/Flask app file."
                )
                self.signals.finished.emit(1)
                return

            # Auto pick if only one candidate; otherwise ask user
            if len(cands) == 1:
                label, argv = cands[0]
                self.signals.log.emit(f"Entrypoint: {label}")
            else:
                labels = [c[0] for c in cands]
                choice = self._ask_user_choice("Pick Entrypoint", "Multiple entrypoints found:", labels)
                if choice is None:
                    self.signals.log.emit("Cancelled.")
                    self.signals.finished.emit(1)
                    return
                idx = labels.index(choice)
                label, argv = cands[idx]
                self.signals.log.emit(f"Entrypoint: {label}")

            self.signals.status.emit("Running project...")
            rc, _ = self._run_and_stream([str(py_exec), *argv], cwd=project_root)
            self.signals.finished.emit(rc)

        except Exception as e:
            self.signals.log.emit(f"Fatal error: {e}")
            self.signals.finished.emit(1)

    def workflow_single_file(self, script: Path) -> None:
        try:
            project_root = script.parent
            venv_path = project_root / VENV_DIRNAME

            self.signals.log.emit(f"Script: {script}")
            self.signals.status.emit("Creating virtual environment...")
            create_venv(venv_path)
            py_exec = venv_python(venv_path)

            retries = 0
            failed_installs = set()

            while retries <= MAX_INSTALL_RETRIES and not self.stop_requested:
                self.signals.status.emit(f"Running script... (attempt {retries+1})")
                rc, out = self._run_and_stream([str(py_exec), str(script)], cwd=project_root)

                if rc == 0 or self.stop_requested:
                    self.signals.finished.emit(rc)
                    return

                missing = parse_missing_module(out)
                if not missing:
                    self.signals.log.emit(f"Script failed (exit code {rc})")
                    self.signals.finished.emit(rc)
                    return

                if missing in failed_installs:
                    self.signals.log.emit(f"Already failed to install '{missing}'. Aborting.")
                    self.signals.finished.emit(rc)
                    return

                self.signals.log.emit(f"Missing module detected: {missing}")
                self.signals.status.emit(f"Installing {missing} ...")
                prc = self._pip_install(py_exec, project_root, ["install", missing])
                if prc != 0:
                    failed_installs.add(missing)
                    self.signals.log.emit(f"pip install {missing} failed.")
                    self.signals.finished.emit(rc)
                    return

                retries += 1

            self.signals.finished.emit(1)

        except Exception as e:
            self.signals.log.emit(f"Fatal error: {e}")
            self.signals.finished.emit(1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PyRunner()
    window.show()
    sys.exit(app.exec())
