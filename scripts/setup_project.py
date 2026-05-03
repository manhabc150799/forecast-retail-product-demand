# tao cay thu muc chuan theo guideline, chay 1 lan la du
# ko can tao bang tay tung folder nua

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# danh sach day du cac folder can tao, bao gom results cho 5 model x 2 phase
DIRECTORIES: list[str] = [
    "src",
    "src/data",
    "src/models",
    "src/evaluation",
    "scripts",
    "results/naive/phase1",
    "results/naive/phase2",
    "results/snaive/phase1",
    "results/snaive/phase2",
    "results/sarimax/phase1",
    "results/sarimax/phase2",
    "results/prophet/phase1",
    "results/prophet/phase2",
    "results/lstm/phase1",
    "results/lstm/phase2",
    "data",
    "notebooks",
    "docs",
    "report",
]

# can __init__.py de python nhan la package, import duoc
PYTHON_PACKAGES: list[str] = [
    "src",
    "src/data",
    "src/models",
    "src/evaluation",
]


def create_directory_tree() -> None:
    for rel in DIRECTORIES:
        dir_path = PROJECT_ROOT / rel
        dir_path.mkdir(parents=True, exist_ok=True)

        # .gitkeep de git track folder rong, khong co thi git bo qua
        gitkeep = dir_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    for pkg in PYTHON_PACKAGES:
        init_file = PROJECT_ROOT / pkg / "__init__.py"
        if not init_file.exists():
            init_file.touch()

    print("[OK] Directory tree created successfully.")


def print_tree(root: Path, prefix: str = "") -> None:
    # in cay thu muc dep, bo cac folder khong lien quan
    ignore = {".git", "__pycache__", ".idea", ".vscode", ".qodo", "out",
              "auxil", "node_modules", ".mypy_cache"}
    entries = sorted(
        [e for e in root.iterdir() if e.name not in ignore],
        key=lambda p: (not p.is_dir(), p.name.lower()),
    )
    for i, entry in enumerate(entries):
        connector = "+-- " if i == len(entries) - 1 else "|-- "
        print(f"{prefix}{connector}{entry.name}")
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "|   "
            print_tree(entry, prefix + extension)


if __name__ == "__main__":
    create_directory_tree()
    print(f"\nProject root: {PROJECT_ROOT}\n")
    print_tree(PROJECT_ROOT)
