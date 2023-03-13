![xron_logo](https://github.com/haotianteng/Xron/blob/master/docs/images/xron_logo.png)
Xron (ˈkairɑn) is a basecaller inherited from [Chiron](https://github.com/haotianteng/Chiron) that could identify methylation modification in Oxford Nanopore sequencer  
Using a deep learning CNN+RNN+CTC structure to establish end-to-end basecalling for the nanopore sequencer.  
Built with **PyTorch** and python 3.8+

<!--
%If you found Xron useful, please consider to cite:  
%Cite paper need to be released
-->


RNA basecall:
```
python xron/xron_eval.py -i <input_fast5_folder> -o <output_folder> --model models/ENEYFT
```

---
## Table of contents

- [Install](#install)
    - [Install using `pip`](#install-using-pip)
    - [Install from GitHub](#install-from-github)
- [Basecall](#basecall)
    - [Test run](#test-run)
    - [Output](#output)
    - [Output format](#output-format)
- [Training](#training)
    - [Hardware request](#hardware-request)
    - [Prepare training data set](#prepare-training-data-set)
    - [Train a model](#train-a-model)

## Install
### <a name="install-using-pip"></a> Install using `pip` (recommended)
To install with `pip`:

```
pip install xron  
```
This will install Xron
PyTorch need to be installed according to your CUDA version and GPU version.

### <a name="install-from-github"></a> Install from Source

```
git clone https://github.com/haotianteng/Xron.git
cd Xron
```
You will also need to install dependencies.
```
python setup.py install
```

## Segmentation using NHMM
Xron also include a non-homegeneous HMM (NHMM) for signal re-sqquigle. To use it:
[TODO] add the code

### Test run

We provided sample code in xron-samples folder to achieve m6A-aware basecall and identify m6A site.
```
python xron/xron_eval.py -i xron/example_folder/ -o <output_folder> -m xron/models/ENEYFT --beam 30 --fast5
```

## Training
### Hardware request
Xron training requires GPU, our training is conducted on Nvidia GeForce 3090Ti
### Prepare training data set
To prepare training dataset for m6A training, we offered scripts to extract training data from basecalled fast5 files. We require a control dataset and a fully/highly methylated dataset.
```bash
python xron/utils/prepare_chunk.py -i $FAST5/ -o $DF/ --extract_seq --write_correction --basecall_entry 001 --alternative_entry 000 --basecaller guppy --reference $REFERENCE_FASTA --mode rna_meth --extract_kmer -k 5 --chunk_len 4000
```
basecall_entry is the basecalled entry in the fast5 files, usually is 000 or 001. --basecaller specify the basecaller used, can be guppy or xron. The script will also write the corrected sequence back to FAST5 files if --write_correction is set. This is vital before we further re-squiggle the reads.

Then we use the NRHMM to re-squiggle the dataset.
```bash
python xron/nrhmm/hmm_relabel.py -i $DF/ -m models/NRHMM/
```
Finally the datasets are hybrid to make the training dataset
```bash
python xron/nrhmm/hybrid_data.py -c $CONTROL/ -m $METH/ -o $OUTPUT/
```

### Train a model

```bash
python xron/xron_train_supervised.py  -i $DF/chunks.npy --seq $DF/seqs.npy --seq_lens.npy $DF/seq_lens.npy -o $OUTPUT/$MODEL_NAME 
```
