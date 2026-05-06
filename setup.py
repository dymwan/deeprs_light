from setuptools import setup, find_packages

setup(
    name="deeprs_light",
    version="0.1.0",
    description="A lightweight standardized testing scaffold for remote sensing CV models",
    author="deeprs",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pycocotools>=2.0.7",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.7.0",
        "rich>=13.0.0",
        "tensorboard>=2.12.0",
        "msgpack>=1.0.0",
        "lmdb>=1.4.0",
    ],
)
