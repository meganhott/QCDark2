from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "rpa_dielectric_cython",
        ["rpa_dielectric_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    ext_modules=cythonize(extensions),
)