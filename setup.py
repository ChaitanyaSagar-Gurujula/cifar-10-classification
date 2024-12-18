from setuptools import setup, find_packages

setup(
    name="mnist-fine-tuned-pro",
    version="0.1",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires=">=3.7",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "matplotlib>=3.5.0",
        "numpy>=1.21.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
    ],
) 