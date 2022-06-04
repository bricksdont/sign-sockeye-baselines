#! /bin/bash

module load volta anaconda3 nvidia/cuda10.2-cudnn7.6.5

scripts=`dirname "$0"`
base=$scripts/../..

venvs=$base/venvs
tools=$base/tools

export TMPDIR="/var/tmp"

mkdir -p $tools

source activate $venvs/sockeye3

# install Sockeye (install custom branch)

# TODO: this currently does not work until  https://github.com/sign-language-processing/datasets version 0.0.7
# is pushed to PyPi

pip --no-cache-dir install git+https://github.com/ZurichNLP/sockeye.git@continuous_outputs_3.1

# install Moses scripts for preprocessing

git clone https://github.com/bricksdont/moses-scripts $tools/moses-scripts

# install BPE library and sentencepiece for subword regularization

pip install subword-nmt sentencepiece

# install tfds SL datasets

pip install --upgrade sign-language-datasets==0.0.7 pose-format==0.0.14

# install subtitle library

pip install srt
