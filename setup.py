from setuptools import setup,find_packages,Extension
from setuptools.command.install import install
import numpy as np
import os
# read the contents of your README file
with open('README.md') as f:
    long_description = f.read()
print(long_description)
class CustomInstallCommand(install):
    def run(self):
        print("\nThis package is licensed under the GNU General Public License v3.0 (GPLv3).")
        print("Please refer to the LICENSE file for more information.\n")
        install.run(self)

install_requires=[
  'h5py',
  'mappy>=2.10.0',
  'numpy==1.24.4',
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
  'torch>=1.12.0',
  'torchvision>=0.13.0',
  'torchaudio>=0.12.0',
  'boostnano',
  'editdistance==0.6.1',
  'boostnano',
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
  keywords = ['basecaller', 'nanopore', 'sequencing','neural network','RNA methylation'], 
  license="GPL 3.0",
  classifiers = ['License :: OSI Approved :: GNU General Public License v3 (GPLv3)'],
  install_requires=install_requires,
  entry_points={'console_scripts':['xron=xron.entry:main'],},
  long_description=long_description,
  include_dirs = [np.get_include()],
  long_description_content_type='text/markdown',
)
