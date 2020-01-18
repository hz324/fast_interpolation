from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["generate_random_spd.pyx"],
                            annotate=True)
    )
