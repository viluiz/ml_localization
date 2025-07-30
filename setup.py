
from setuptools import setup, find_packages

setup(
    name="ml_localization",
    version="0.1.0",
    description="Implementation of the method machine learning localization",
    author="Vinicius Luiz Santos Silva",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "lightgbm"
    ],
    entry_points={
        "console_scripts": [
            "ml_localization=ml_localization.cli:main"
        ]
    }
)

