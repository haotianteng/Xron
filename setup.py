from setuptools import setup,find_packages,Extension
import numpy as np
import os
# read the contents of your README file
with open('README.md') as f:
    long_description = f.read()
print(long_description)

print("""
*******************************************************************
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
(c) 2023 Haotian Teng
*******************************************************************
""")

install_requires=[
  'h5py>=2.7.0',
  'mappy>=2.10.0',
  'numpy>=1.13.3',
  'statsmodels>=0.8.0',
  'tqdm>=4.23.0',
  'scipy>=1.0.1',
  'biopython==1.73',
  'google-auth==2.18.1',
  'oauthlib==3.2.2',
  'packaging>=18.0',
  'ont-fast5-api>=0.3.1',
  'wget>=3.2',
  'pysam>=0.21.0',
  'tensorboard',
  'matplotlib',
  'seaborn',
  'pandas',
  'toml',
  'fast-ctc-decode',
  'editdistance>=0.5.3',
  'torch==1.12.0',
  'torchvision==0.13.0',
  'torchaudio==0.12.0'
]
exec(open('xron/_version.py').read()) #readount the __version__ variable
setup(
  name = 'xron',
  packages = find_packages(exclude=["*.test", "*test.*", "test.*", "test"]),
  version = __version__,
  include_package_data=True,
  description = 'A deep neural network basecaller for nanopore sequencing.',
  author = 'Haotian Teng',
  author_email = 'havens.teng@gmail.com',
  url = 'https://github.com/haotianteng/Xron', 
  download_url = 'https://github.com/haotianteng/Xron/archive/1.0.0.tar.gz', 
  keywords = ['basecaller', 'nanopore', 'sequencing','neural network'], 
  license="MPL 2.0",
  classifiers = ['License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)'],
  install_requires=install_requires,
  entry_points={'console_scripts':['xron=xron.entry:main'],},
  long_description=long_description,
  ext_modules = [ Extension('boostnano.hmm', sources = ['boostnano/hmm.cpp'],extra_compile_args=['-std=c++11'])],
  include_dirs = [np.get_include()],
  long_description_content_type='text/markdown',
)
