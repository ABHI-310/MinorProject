"""One entry point for maintained app, data, training and verification commands."""
import argparse
import runpy
import subprocess
import sys
from _bootstrap import ROOT

COMMANDS = {
    "app": "run_project.py",
    "verify-api": "verify_backend.py",
    "verify-rul": "validate_rul.py",
    "verify-layout": "verify_layout.py",
    "audit": "revamp.audit",
    "prepare": "revamp.prepare",
    "package": "revamp.package",
    "notebook": "revamp.build_resume_notebook",
    "smoke": "revamp.smoke",
    "pilot": "revamp.pilot",
    "train-source": "revamp.train_source",
    "train-full-boards": "revamp.train_full_boards",
    "full-boards-notebook": "revamp.build_full_boards_notebook",
    "source-notebook": "revamp.build_source_notebook",
    "pilot-notebook": "revamp.build_pilot_notebook",
    "test": None,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "test":
        raise SystemExit(subprocess.call([sys.executable,"-m","pytest",*(args.arguments or ["tests","-q"])],cwd=ROOT))
    target = COMMANDS[args.command]
    sys.argv = [target,*args.arguments]
    if target.endswith(".py"):
        runpy.run_path(str(ROOT/"scripts"/target),run_name="__main__")
    else:
        runpy.run_module(target,run_name="__main__")

if __name__ == "__main__":
    main()
