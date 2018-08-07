from setuptools import find_packages
from setuptools import setup

PYTEST_VERSION_ = '3.3.0'

setup(name='evalmate',
      version='1.0.0',
      description='Evalmate is a set of tools for evaluate audio related machine learning tasks.',
      long_description='Evalmate is a set of tools for evaluate audio related machine learning tasks.',
      url='https://github.com/ynop/evalmate',
      download_url='https://github.com/ynop/evalmate/releases',
      author='Matthias Buechi, Andreas Ahlenstorf',
      author_email='buec@zhaw.ch',
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3 :: Only'
      ],
      keywords='audio music evaluation metrics',
      license='MIT',
      packages=find_packages(exclude=['tests']),
      install_requires=[
          'audioread >= 1.0.0',
          'numpy >= 1.14.0',
          'scipy >= 1.1.0',
      ],
      include_package_data=True,
      zip_safe=False,
      test_suite='tests',
      extras_require={
          'dev': [
              'click==6.7',
              'pytest==%s' % (PYTEST_VERSION_,),
              'pytest-runner==3.0',
              'pytest-cov==2.5.1',
              'Sphinx==1.6.5',
              'sphinx-rtd-theme==0.2.5b1'
          ],
          'ci': ['flake8==3.5.0', 'flake8-quotes==0.12.1'],
      },
      setup_requires=['pytest-runner'],
      tests_require=[
          'pytest==%s' % (PYTEST_VERSION_,)
      ],
      entry_points={
      }
      )
