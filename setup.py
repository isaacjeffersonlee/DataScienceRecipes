from setuptools import setup, find_packages

setup(
    name="recipes",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "tqdm",
        "scikit-learn",  # Only for testing
    ],
)
