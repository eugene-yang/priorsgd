## setup script for compiling the modified version of SGDClassifier
import os
from os.path import join

import setuptools 
from setuptools import setup

import numpy
from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info


# from sklearn._build_utils.get_blas_info
def get_blas_info():
    def atlas_not_found(blas_info_):
        def_macros = blas_info.get('define_macros', [])
        for x in def_macros:
            if x[0] == "NO_ATLAS_INFO":
                # if x[1] != 1 we should have lapack
                # how do we do that now?
                return True
            if x[0] == "ATLAS_INFO":
                if "None" in x[1]:
                    # this one turned up on FreeBSD
                    return True
        return False

    blas_info = get_info('blas_opt', 0)
    if (not blas_info) or atlas_not_found(blas_info):
        cblas_libs = ['cblas']
        blas_info.pop('libraries', None)
    else:
        cblas_libs = blas_info.pop('libraries', [])

    return cblas_libs, blas_info


def configuration():

    config = Configuration('priorsgd', parent_package='', top_path='')

    cblas_libs, blas_info = get_blas_info()
    cblas_includes = [join('priorsgd', 'src', 'cblas'),
                      numpy.get_include(),
                      blas_info.pop('include_dirs', [])]

    libraries = []
    if os.name == 'posix':
        libraries.append('m')
        cblas_libs.append('m')
    
    config.add_extension('seq_dataset',
                         sources=join('priorsgd', 'seq_dataset.pyx'),
                         include_dirs=[numpy.get_include()])

    config.add_extension('weight_vector',
                         sources=join('priorsgd', 'weight_vector.pyx'),
                         include_dirs=cblas_includes,
                         libraries=cblas_libs,
                         **blas_info)
    
    config.add_extension('sgd_fast',
                         sources=join('priorsgd', 'sgd_fast.pyx'),
                         include_dirs=cblas_includes,
                         libraries=cblas_libs,
                         extra_compile_args=blas_info.pop('extra_compile_args',
                                                          []),
                         **blas_info)

    return config

if __name__ == '__main__':
    config = configuration()
    setup(name=config.name,
          version="0.0.1",
          author="Eugene Yang",
          author_email="eugene@ir.cs.georgetown.edu",
          description="Stochastic Gradient Descent with Priors (priorsgd)",
          long_description=open('./README.md').read(),
          long_description_content_type="text/markdown",
          url="https://github.com/eugene-yang/priorsgd",
          setup_requires=['setuptools>=18.0',
                          'numpy>=1.14.0', 
                          'cython>=0.27.3'],
          install_requires=['scipy>=1.1.0', 'scikit-learn>=0.19.1'],
          packages=setuptools.find_packages(),
          ext_modules=config.ext_modules)