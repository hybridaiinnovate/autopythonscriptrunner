💤 PyRunner — For Developers Who Refuse to Manually Set Up Environments

Drag. Drop. Run.
No venv drama. No dependency rituals. No framework guesswork.

PyRunner is a zero-effort Python project launcher built for developers who believe:

“If I have to activate one more virtual environment, I’m quitting.”

“Why is this repo not running?”

“It worked on their machine.”

🚀 What It Does

Just drag a project folder into PyRunner and it will:

🔍 Detect the project root

🧪 Create a virtual environment automatically

📦 Install dependencies (requirements.txt, pyproject.toml, setup.cfg)

🧠 Detect the framework

▶️ Run the correct entrypoint

📡 Stream logs live

🛑 Let you stop it instantly

You literally don’t have to think.

🧠 Smart Framework Detection

Supports automatic detection for:

⚡ FastAPI / Uvicorn

Detects FastAPI() apps

Runs uvicorn module:app

Supports --reload

🐍 Django

Detects manage.py

Offers:

runserver

check

test

🔥 Flask

Detects Flask() apps

Runs via flask run --app module:app

🧰 CLI Tools

Reads:

pyproject.toml ([project.scripts])

setup.cfg (console_scripts)

Runs CLI entrypoints automatically

📦 Python Packages

Detects __main__.py

Runs via python -m package

📝 General Scripts

Detects main.py, app.py, run.py, server.py

Falls back to root .py files

🎯 Designed For

Developers who hate manual setup

People testing random GitHub repos

Lazy backend engineers

Students

Framework hoppers

“Why doesn’t this run?” moments

💻 How To Use
pip install PySide6
python pyrunner.py


Then:

Drag a project folder in

Or click “Select Project Folder”

Or run a single .py file

Done.

🛑 What You Don’t Have To Do

❌ Create venv

❌ Activate venv

❌ Install dependencies manually

❌ Figure out entrypoint

❌ Guess the framework

❌ Read README just to run it

🧠 Philosophy

If a Python project exists…

It should run.

No ceremony.

No suffering.

⚠️ Warning

May increase laziness.
May reduce DevOps knowledge.
May make you allergic to manual setup.

Use responsibly or dont I really don't care....

If you'd like, I can also write:

A more serious enterprise version

A chaotic meme-heavy version

A README.md full version (with badges)

A Hacker News–bait description

A Product Hunt launch description
