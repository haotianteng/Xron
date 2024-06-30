![xron_logo](https://github.com/haotianteng/Xron/blob/master/docs/images/xron_logo.png)
Xron (ˈkairɑn) is a methylation basecaller that could identify m6A methylation modification from ONT direct RNA sequencing.  
Using a deep learning CNN+RNN+CTC structure to establish end-to-end basecalling for the nanopore sequencer.  
The name is inherited from [Chiron](https://github.com/haotianteng/Chiron)
Built with **PyTorch** and python 3.8+

<!--
%If you found Xron useful, please consider to cite:  
%Cite paper need to be released
-->


m6A-aware RNA basecall one-liner:
```
xron call -i <input_fast5_folder> -o <output_folder> -m models/ENEYFT --boostnano
```

---
## Table of contents

- [Table of contents](#table-of-contents)
- [Install](#install)
  - [Install from Source](#install-from-source)
  - [Install from Pypi](#install-from-pypi)
- [Basecall](#basecall)
- [Segmentation using NHMM](#segmentation-using-nhmm)
  - [Prepare chunk dataset](#prepare-chunk-dataset)
  - [Realign the signal using NHMM.](#realign-the-signal-using-nhmm)
- [Training](#training)
  
## Install
For either installation method, recommend to create a vritual environment first using conda or venv, take conda for example
```bash
conda create --name YOUR_VIRTUAL_ENVIRONMENT python=3.8
conda activate YOUR_VIRTUAL_ENVIRONMENT
```
Then you can install from our pypi repository or install the newest version from github repository.
### Install from Source

```
git clone --recursive https://github.com/haotianteng/Xron.git
cd Xron
python setup.py install
```
You will also need to PyTorch according to your environment. PyTorch 1.12 is the version we used when developing xron, but later version is usually also good.

### Install from Pypi
```bash
pip install xron
```
## Basecall
Before running basecall using Xron, you need to download the models from our AWS s3 bucket by running **xron init**
```bash
xron init
```
This will automatically download the models and put them into the *models* folder.
We provided sample code in xron-samples folder to achieve m6A-aware basecall and identify m6A site. 
To run xron on raw fast5 files:
```
xron call -i ${INPUT_FAST5} -o ${OUTPUT} -m models/ENEYFT --fast5 --beam 50 --chunk_len 2000
```

## Segmentation using NHMM
### Prepare chunk dataset
Xron also include a non-homegeneous HMM (NHMM) for signal re-sqquigle. To use it:
Firstly we need to extract the chunk and basecalled sequence using **prepare** module
```bash
xron prepare -i ${FAST5_FOLDER} -o ${CHUNK_FOLDER} --extract_seq --basecaller guppy --reference ${REFERENCE} --mode rna_meth --extract_kmer -k 5 --chunk_len 4000 --write_correction
```
Replace the FAST5_FOLDER, CHUNK_FOLDER and REFERENCE with your basecalled fast5 file folder, your output folder and the path to the reference genome fasta file.

### Realign the signal using NHMM.
Then run the NHMM to realign ("resquiggle") the signal.
```bash
xron relabel -i ${CHUNK_FOLDER} -m ${MODEL} --device $DEVICE
```
This will generate a paths.py file under CHUNK_FOLDER which gives the kmer segmentation of the chunks.

## Training
To train a new Xron model using your own dataset, you need to prepare your own training dataset, the dataset should includes a signal file (chunks.npy), labelled sequences (seqs.npy) and sequence length for each read (seq_lens.npy), and then run the xron supervised training module
```bash
xron train -i chunks.npy --seq seqs.npy --seq_len seq_lens.npy --model_folder OUTPUT_MODEL_FOLDER
```
Training Xron model from scratch is hard, I would recommend to fine-tune our model by specify --load flag, for example we can finetune the provided ENEYFT model (model trained using cross-linked ENE dataset and finetuned on Yeast dataset):
```bash
xron train -i chunks.npy --seq seqs.npy --seq_len seq_lens.npy --model_folder models/ENEYFT --load
```

