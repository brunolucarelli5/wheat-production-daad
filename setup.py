"""
Setup para instalar el proyecto como paquete Python (desarrollo).
"""
from setuptools import find_packages, setup

setup(
    name="daad-trigo",
    version="0.1.0",
    description="Predicción de rendimiento de trigo con ML (Beca UTN-DAAD)",
    author="UTN-DAAD",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=2.4.0",
        "pandas>=3.0.0",
        "scikit-learn>=1.8.0",
        "matplotlib>=3.10.0",
        "shap>=0.50.0",
        "requests>=2.32.0",
    ],
)
