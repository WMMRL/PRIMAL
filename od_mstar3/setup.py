from distutils.core import Extension, setup

from Cython.Build import cythonize

setup(ext_modules = cythonize(Extension(
           "cpp_mstar",                                
           sources=["cython_od_mstar.pyx"], 
           extra_compile_args=["-std=c++11", "-I/usr/local/include"]
      )))
