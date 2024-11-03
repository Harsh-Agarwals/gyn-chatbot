import os

app_folder = "gyn_genai"

files = [
    f"{app_folder}/__init__.py",
    f"{app_folder}/config",
    f"{app_folder}/config/__init__.py",
    f"{app_folder}/config/config.py",
    f"{app_folder}/config/utils.py",
    f"{app_folder}/data",
    f"{app_folder}/data/__init__.py",
    f"{app_folder}/data/data_loader.py",
    f"{app_folder}/data/preprocessing.py",
    f"{app_folder}/api",
    f"{app_folder}/api/__init__.py",
    f"{app_folder}/api/api_endpoints.py",
    f"{app_folder}/models",
    f"{app_folder}/models/__init__.py",
    f"{app_folder}/models/model.py",
    f"{app_folder}/models/tokenizer.py",

    "tests/__init__.py",
    "tests/test_app.py",
    "tests/test_data.py",
    "tests/test_model.py",

    "main.py"
]