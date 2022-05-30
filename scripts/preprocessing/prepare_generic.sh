#! /bin/bash

# calling script has to set:

# $base
# $src
# $trg
# $model_name
# $seed

base=$1
src=$2
trg=$3
model_name=$4
seed=$5

# measure time

SECONDS=0

venvs=$base/venvs

eval "$(conda shell.bash hook)"
source activate $venvs/sockeye3

# after ativating the env on purpose

set -u

data=$base/data
data_sub=$data/${src}-${trg}
data_sub_sub=$data_sub/$model_name

prepared=$base/prepared
prepared_sub=$prepared/${src}-${trg}
prepared_sub_sub=$prepared_sub/$model_name

if [[ -d $prepared_sub_sub ]]; then
    echo "prepared_sub_sub already exists: $prepared_sub_sub"
    echo "Skipping. Delete files to repeat step."
    exit 0
fi

mkdir -p $prepared_sub_sub

cmd="python -m sockeye.prepare_data -s $data_sub_sub/train.h5 -t $data_sub_sub/train.pieces.trg -o $prepared_sub_sub --max-seq-len 500:250 --seed $seed"

echo "Executing:"
echo "$cmd"

python -m sockeye.prepare_data \
                        -s $data_sub_sub/train.h5 \
                        -t $data_sub_sub/train.pieces.trg \
                        -o $prepared_sub_sub \
                        --max-seq-len 500:250 \
                        --seed $seed \
                        --shared-vocab

echo "time taken:"
echo "$SECONDS seconds"