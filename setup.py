from setuptools import setup, find_packages

setup(
    name="rwkv-ss",
    version="0.1.0",
    description="RWKV-v7 TF-domain speech separation",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # Loose bounds here — pinned versions for reproducible installs
    # live in requirements.txt, which is what Docker images use.
    install_requires=[
        "soundfile>=0.12",
        "numpy>=1.24",
        "tqdm>=4.65",
        "tensorboard>=2.14",
        "torch>=2.0.0",
    ],
    extras_require={
        # pip install rwkv-ss[eval]
        "eval": [
            "pesq>=0.0.4",
            "pystoi>=0.4.1",
            "scipy>=1.10",
            "torchmetrics>=1.0",
        ],
        # pip install rwkv-ss[dev]
        "dev": [
            "pytest>=7.4",
        ],
    },
)
