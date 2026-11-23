from pathlib import Path

def get_rootdir():
    path = Path(__file__).resolve()
    for folder in path.parents:
       # pyproject.toml only exists in root
       if (folder / "pyproject.toml").exists():
           return folder
    raise FileNotFoundError(f"no pyproject.toml found above {path}")