import os
from setuptools import setup

setup(
    name="sabgbp",
    package_dir = {
        'sabgbp_utils': os.path.join(".", "utils"),
        'sabgbp_models': os.path.join(".", "models"),
        'sabgbp_configs': os.path.join(".", "configs"),
        'sabgbp_datasets': os.path.join(".", "datasets"),
    },
    version="0.0.1",
    python_requires=">=3.6",
    packages=[
        "sabgbp_utils",
        "sabgbp_models",
        "sabgbp_configs",
        "sabgbp_datasets",
    ],
    package_data={

    }
)
