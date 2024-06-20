from setuptools import setup

setup(
    name="sabgbp",
    package_dir = {'sabgbp': '.'},
    version="0.0.1",
    python_requires=">=3.6",
    packages=[
        "sabgbp",
        "sabgbp.utils",
        "sabgbp.models",
        "sabgbp.datasets",
    ],
    package_data={
        "sabgbp": [

        ]
    }
)
