from setuptools import setup, find_packages

setup(
    name="batd",
    version="0.1.0",
    description="Backdoor Attacks on Tabular Data",
    author="BATD Team",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "xgboost",
        "catboost",
    ],
    python_requires=">=3.11",
) 