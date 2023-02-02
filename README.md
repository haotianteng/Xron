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
python xron/xron_call.py -i <input_fast5_folder> -o <output_folder> -c m6a_model.config
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
    - [Training parameters](#training-parameters)

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
python chiron/entry.py call -i chiron/example_folder/ -o <output_folder> -m chiron/model/DNA_default --preset dna-pre
```

```
python xron/xron_rcnn_train.py  --data_dir <signal_label folder/ tfrecord file> --log_dir <model_log>
```
### Training parameters
Following parameters can be passed to Chiron when training

`data_dir`(Required): The folder containing your signal and label files.  
`log_dir`(Required): The folder where you want to save the model.  
`model_name`(Required): The name of the model. The record will be stored in the directory `log_dir/model_name/`
`sequence_len`: The length of the segment you want to separate the sequence into. Longer length requires larger RAM.  
`batch_size`: The batch size.  
`step_rate`: Learning rate of the optimizer.  
`max_step`: Maximum step of the optimizer.  
`k_mer`: Chiron supports learning based on k-mer instead of a single nucleotide, this should be an odd number, even numbers will cause an error.  
`retrain`: If this is a new model, or you want to load the model you trained before. The model will be loaded from  `log_dir/model_name/`  
