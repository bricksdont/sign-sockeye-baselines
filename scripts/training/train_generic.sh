#!/bin/bash

# calling script needs to set:

# $base
# $src
# $trg
# $model_name
# $dry_run
# $seed
# $pose_type
# $bucket_scaling
# $max_seq_len_source
# $bucket_width
# $initial_learning_rate
# $dropout
# $num_layers_encoder
# $num_layers_decoder

base=$1
src=$2
trg=$3
model_name=$4
dry_run=$5
seed=$6
pose_type=$7
bucket_scaling=$8
max_seq_len_source=$9
bucket_width=${10}
initial_learning_rate=${11}
dropout=${12}
num_layers_encoder=${13}
num_layers_decoder=${14}

venvs=$base/venvs

data=$base/data
data_sub=$data/${src}-${trg}
data_sub_sub=$data_sub/$model_name

prepared=$base/prepared
prepared_sub=$prepared/${src}-${trg}
prepared_sub_sub=$prepared_sub/$model_name

models=$base/models
models_sub=$models/${src}-${trg}
models_sub_sub=$models_sub/$model_name

mkdir -p $models_sub_sub

eval "$(conda shell.bash hook)"
source activate $venvs/sockeye3

# after activating the env on purpose

set -u

# parameters are the same for all Transformer models

num_embed="512:512"
num_layers="$num_layers_encoder:$num_layers_decoder"
transformer_model_size="512"
transformer_attention_heads="8"
transformer_feed_forward_num_hidden="2048"

embed_dropout=$dropout
fc_embed_dropout_pre=$dropout
fc_embed_dropout_post=$dropout
transformer_dropout=$dropout
batch_size=1024
decode_and_evaluate=500
checkpoint_interval=1000

# check if training is finished

if [[ -f $models_sub_sub/log ]]; then

    training_finished=`grep "Training finished" $models_sub_sub/log | wc -l`

    if [[ $training_finished != 0 ]]; then
        echo "Training is finished"
        echo "Skipping. Delete files to repeat step."
        exit 0
    fi
fi

if [[ $dry_run == "true" ]]; then
    dry_run_additional_args="--max-updates 10 --use-cpu"
    checkpoint_interval=10
    num_embed="64:64"
    num_layers="6:6"
    transformer_model_size="64"
    transformer_attention_heads="1"
    transformer_feed_forward_num_hidden="64"
else
    dry_run_additional_args=""
fi

if [[ $pose_type == "openpose" ]]; then
    num_features=270
else
    # mediapipe poses
    num_features=609
fi

if [[ $bucket_scaling == "true" ]]; then
    bucket_scaling_arg="--bucket-scaling"
else
    bucket_scaling_arg=""
fi

##################################################

python -m sockeye.train \
-d $prepared_sub_sub \
-vs $data_sub_sub/dev.src \
-vt $data_sub_sub/dev.pieces.trg \
--output $models_sub_sub \
--seed $seed \
--batch-type word \
--batch-size $batch_size \
--bucket-width $bucket_width \
--device-id 0 \
--source-is-continuous \
--source-continuous-num-features $num_features \
--encoder transformer \
--decoder transformer \
--num-layers $num_layers \
--transformer-model-size $transformer_model_size \
--transformer-attention-heads $transformer_attention_heads \
--transformer-feed-forward-num-hidden $transformer_feed_forward_num_hidden \
--transformer-preprocess n \
--transformer-postprocess dr \
--transformer-dropout-attention $transformer_dropout \
--transformer-dropout-act $transformer_dropout \
--transformer-dropout-prepost $transformer_dropout \
--transformer-positional-embedding-type fixed \
--embed-dropout 0.0:$embed_dropout \
--fc-embed-dropout-pre $fc_embed_dropout_pre \
--fc-embed-dropout-post $fc_embed_dropout_post \
--weight-tying-type trg_softmax \
--num-embed $num_embed \
--num-words 64000:64000 \
--optimizer adam \
--initial-learning-rate $initial_learning_rate \
--learning-rate-reduce-num-not-improved 4 \
--checkpoint-interval $checkpoint_interval \
--keep-last-params 30 \
--max-seq-len $max_seq_len_source:250 \
--learning-rate-reduce-factor 0.7 \
--decode-and-evaluate $decode_and_evaluate \
--max-num-checkpoint-not-improved 10 \
--min-num-epochs 0 \
--gradient-clipping-type abs \
--gradient-clipping-threshold 1 $bucket_scaling_arg $dry_run_additional_args
