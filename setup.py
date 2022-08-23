#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='applesoss',
      version='0.1.0',
      license='MIT',
      author='Michael Radica',
      author_email='michael.radica@umontreal.ca',
      packages=['applesoss'],
      include_package_data=True,
      url='https://github.com/radicamc/APPLESOSS',
      description='A Producer of ProfiLEs for SOSS',
      package_data={'': ['README.md', 'LICENSE']},
      install_requires=['astropy', 'matplotlib', 'numpy', 'scipy', 'webbpsf'],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        ],
      )
