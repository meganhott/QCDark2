from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(name='qcdark2', 
      version='2.0.0', 
      long_description=long_description, 
      long_description_content_type='text/markdown',
      url='https://github.com/meganhott/QCDark2',
      author='Megan Hott',
      author_email='megan.hott@stonybrook.edu',
      packages=find_packages('.'),
      install_requires=['numpy', ' numba', 'pyscf', 'h5py', 'scipy', 'psutil'],
      python_requires='>=3.9',
      )