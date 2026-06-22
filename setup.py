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
        "argbind>=0.3.7",
        "descript-audiotools>=0.7.2",
        "einops",
        "numpy>=1.24",
        "numba>=0.5.7",
        "PyYAML>=6.0",
        "soundfile>=0.12",
        "tensorboard>=2.14",
        "torch>=2.0.0",
        "torchaudio",
        "tqdm>=4.65",
        "mir_eval>=0.7",
        "flash-linear-attention",
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
            "pytest-cov",
            "pynvml",
            "psutil",
            "pandas",
            "onnx",
            "onnx-simplifier",
            "seaborn",
            "jupyterlab",
            "watchdog",
            "pesq",
            "tabulate",
            "encodec",
        ],
    },
)
