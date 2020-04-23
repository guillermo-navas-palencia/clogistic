import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# install requirements
install_requires = [
    'cvxpy>=1.0.31',
    'numpy',
    'scipy',
    'scikit-learn>=0.20.0'
]


description = '''
Logistic regression with bound and linear constraints. L1, L2, SOS and
Elastic-Net regularization.
'''


setup(
    name='clogistic',
    version='0.1.0',
    description=description,
    long_description=read('README.rst'),
    author="Guillermo Navas-Palencia",
    author_email="g.navas.palencia@gmail.com",
    url="https://github.com/guillermo-navas-palencia/clogistic",
    packages=['clogistic'],
    platforms="any",
    include_package_data=True,
    license="Apache Licence 2.0",
    python_requires='>=3.6',
    install_requires=install_requires,
    tests_require=['pytest'],
    classifiers=[
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3'],
    keywords='machine-learning, logistic-regression, statistics, data-science'
)
