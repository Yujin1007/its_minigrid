from setuptools import setup, find_packages

setup(
    name="grid_navigation_rl",
    version="1.0.0",
    author="Yujin Kim",
    author_email="yk826@cornell.edu",
    description="Reinforcement Learning for Grid for Iterative teacher-student policy distillation",
    packages=find_packages(),  # Automatically find all packages in the directory
    py_modules=[
        "__init__",
        "grid_nav",
        "gridnav_rl_callbacks",
        "reinforce_model",
        "rl_with_seq_matching",
        "run_seq_matching_on_examples",
        "toy_env_utils",
    ],
    install_requires=[
        "gymnasium>=0.26.0",
        "torch>=1.10.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "stable-baselines3>=1.6.2",
        "wandb>=0.15.0",
        "Pillow>=8.0.0",
        "tqdm>=4.60.0",
        "loguru>=0.6.0",
        "omegaconf>=2.2.3",
        "hydra-core>=1.1.0",
        "imageio>=2.9.0",
    ],
    entry_points={
        "console_scripts": [
            "run_seq_matching=run_seq_matching_on_examples:main",  # Adjust `main` if necessary
        ]
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)