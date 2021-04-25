"""Install command: pip3 install -e ."""

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = []

setup(name='chexpert',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      description='chexpert transfer learning self-supervised',
      url='http://github.com/tomergolany/self_supervised',
      author='Tomer Golany',
      author_email='tomer.golany@gmail.com',
      license='Technion',
      packages=find_packages(),
      include_package_data=False,
      zip_safe=False)
