import os
from pathlib import Path

files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    "app.py",
    "setup.py"
]

for filepath in files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath) or (os.path.getsize(filepath)==0)):
        with open(filepath, 'w') as f:
            pass