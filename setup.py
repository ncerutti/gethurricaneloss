# setup.py

from setuptools import setup, find_packages

setup(
    name="gethurricaneloss",
    version="0.1",
    packages=find_packages(),
    url="https://github.com/ncerutti/gethurricaneloss",
    install_requires=[
        "numba==0.58.0",
        "numpy==1.25.0",
        "pytest==7.4.2",
        "tqdm==4.66.3",
    ],
    entry_points={
        "console_scripts": [
            "gethurricaneloss_base = src.gethurricaneloss_base:main",
            "gethurricaneloss_jit = src.gethurricaneloss_jit:main",
            "gethurricaneloss_mp = src.gethurricaneloss_mp:main",
            "gethurricaneloss_mp_para = src.gethurricaneloss_mp_para:main",
            "gethurricaneloss_two = src.gethurricaneloss_two:main",
            "gethurricaneloss = src.gethurricaneloss_two:main",
        ],
    },
)
