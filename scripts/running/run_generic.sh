#! /bin/bash

# calling process needs to set:
# $base
# $src
# $trg
# $model_name
#
# optional:
# $dry_run (values: "true" or "false")
# $training_corpora
# $testing_corpora
# $seed
# $pose_type
# $sentencepiece_vocab_size
# $force_target_fps
# $normalize_poses
# $bucket_scaling
# $local_download_data
# $$max_seq_len_source
# $bucket_width
# $initial_learning_rate
# $dropout
# $num_layers_encoder
# $num_layers_decoder
#
# optional environment variables to be set when calling a run script (these are private tokens that should not appear
# in logs or commits):
# $ZENODO_TOKEN_FOCUSNEWS
# $ZENODO_TOKEN_SRF_POSES
# $ZENODO_TOKEN_SRF_VIDEOS_SUBTITLES

module load volta nvidia/cuda10.2-cudnn7.6.5 anaconda3

scripts=$base/scripts
logs=$base/logs

source activate $base/venvs/sockeye3

logs_sub=$logs/${src}-${trg}
logs_sub_sub=$logs_sub/$model_name

SLURM_DEFAULT_FILE_PATTERN="slurm-%j.out"
SLURM_LOG_ARGS="-o $logs_sub_sub/$SLURM_DEFAULT_FILE_PATTERN -e $logs_sub_sub/$SLURM_DEFAULT_FILE_PATTERN"

mkdir -p $logs_sub_sub

# if variables are undefined, set to avoid confusion

if [ -z "$dry_run" ]; then
    dry_run="false"
fi

if [ -z "$local_download_data" ]; then
    local_download_data="false"
fi

if [ -z "$training_corpora" ]; then
    training_corpora="focusnews"
fi

if [ -z "$testing_corpora" ]; then
    testing_corpora="test"
fi

if [ -z "$pose_type" ]; then
    pose_type="openpose"
fi

if [ -z "$seed" ]; then
    seed="1"
fi

if [ -z "$sentencepiece_vocab_size" ]; then
    sentencepiece_vocab_size="1000"
fi

if [ -z "$force_target_fps" ]; then
    force_target_fps="false"
fi

if [ -z "$normalize_poses" ]; then
    normalize_poses="true"
fi

if [ -z "$bucket_scaling" ]; then
    bucket_scaling="true"
fi

if [ -z "$bucket_width" ]; then
    bucket_width=16
fi

if [ -z "$initial_learning_rate" ]; then
    initial_learning_rate="0.0001"
fi

if [ -z "$dropout" ]; then
    dropout="0.1"
fi

if [ -z "$num_layers_encoder" ]; then
    num_layers_encoder=6
fi

if [ -z "$num_layers_decoder" ]; then
    num_layers_decoder=6
fi

if [ -z "$max_seq_len_source" ]; then
    max_seq_len_source=500
fi

# special consideration to Zenodo tokens
# (these must be set as environment variables before / when calling a run script)

if [ -z "$ZENODO_TOKEN_FOCUSNEWS" ]; then
    zenodo_token_focusnews="none"
else
    zenodo_token_focusnews=$ZENODO_TOKEN_FOCUSNEWS
fi

if [ -z "$ZENODO_TOKEN_SRF_POSES" ]; then
    zenodo_token_srf_poses="none"
else
    zenodo_token_srf_poses=$ZENODO_TOKEN_SRF_POSES
fi

if [ -z "$ZENODO_TOKEN_SRF_VIDEOS_SUBTITLES" ]; then
    zenodo_token_srf_videos_subtitles="none"
else
    zenodo_token_srf_videos_subtitles=$ZENODO_TOKEN_SRF_VIDEOS_SUBTITLES
fi

# after setting unset variables: fail if variables are still undefined

set -u

# SLURM job args

DRY_RUN_SLURM_ARGS="--cpus-per-task=2 --time=02:00:00 --mem=16G --partition=generic"

SLURM_ARGS_GENERIC="--cpus-per-task=2 --time=24:00:00 --mem=16G --partition=generic"
SLURM_ARGS_VOLTA_TRAIN="--qos=vesta --time=36:00:00 --gres gpu:Tesla-V100-32GB:1 --cpus-per-task 1 --mem 16g"
SLURM_ARGS_VOLTA_TRANSLATE="--qos=vesta --time=12:00:00 --gres gpu:Tesla-V100-32GB:1 --cpus-per-task 1 --mem 16g"

# if dry run, then all args use generic instances

if [[ $dry_run == "true" ]]; then
  SLURM_ARGS_GENERIC=$DRY_RUN_SLURM_ARGS
  SLURM_ARGS_VOLTA_TRAIN=$DRY_RUN_SLURM_ARGS
  SLURM_ARGS_VOLTA_TRANSLATE=$DRY_RUN_SLURM_ARGS
fi

# log key info

echo "##############################################" | tee -a $logs_sub_sub/MAIN
date | tee -a $logs_sub_sub/MAIN
echo "##############################################" | tee -a $logs_sub_sub/MAIN
echo "LANGPAIR: ${src}-${trg}" | tee -a $logs_sub_sub/MAIN
echo "MODEL NAME: $model_name" | tee -a $logs_sub_sub/MAIN
echo "LOCAL_DOWNLOAD_DATA: $local_download_data" | tee -a $logs_sub_sub/MAIN
echo "TRAINING CORPORA: $training_corpora" | tee -a $logs_sub_sub/MAIN
echo "TESTING CORPORA: $testing_corpora" | tee -a $logs_sub_sub/MAIN
echo "SEED: $seed" | tee -a $logs_sub_sub/MAIN
echo "POSE_TYPE: $pose_type" | tee -a $logs_sub_sub/MAIN
echo "SENTENCEPIECE_VOCAB_SIZE: $sentencepiece_vocab_size" | tee -a $logs_sub_sub/MAIN
echo "FORCE_TARGET_FPS: $force_target_fps" | tee -a $logs_sub_sub/MAIN
echo "NORMALIZE_POSES: $normalize_poses" | tee -a $logs_sub_sub/MAIN
echo "BUCKET SCALING: $bucket_scaling" | tee -a $logs_sub_sub/MAIN
echo "BUCKET_WIDTH: $bucket_width" | tee -a $logs_sub_sub/MAIN
echo "MAX_SEQ_LEN_SOURCE: $max_seq_len_source" | tee -a $logs_sub_sub/MAIN
echo "INITIAL_LEARNING_RATE: $initial_learning_rate" | tee -a $logs_sub_sub/MAIN
echo "DROPOUT: $dropout" | tee -a $logs_sub_sub/MAIN
echo "NUM_LAYERS_ENCODER: $num_layers_encoder" | tee -a $logs_sub_sub/MAIN
echo "NUM_LAYERS_DECODER: $num_layers_decoder" | tee -a $logs_sub_sub/MAIN

echo "DRY RUN: $dry_run" | tee -a $logs_sub_sub/MAIN

# download corpora

id_download=$(
    $scripts/running/sbatch_bare.sh \
    $SLURM_ARGS_GENERIC \
    $SLURM_LOG_ARGS \
    $scripts/downloading/download_generic.sh \
    $base "$training_corpora" $local_download_data "$testing_corpora" \
    $zenodo_token_focusnews $zenodo_token_srf_poses $zenodo_token_srf_videos_subtitles
)

echo "  id_download: $id_download | $logs_sub_sub/slurm-$id_download.out" | tee -a $logs_sub_sub/MAIN

# preprocess: Combine datasets, hold out data, normalize, SPM (depends on download)

id_preprocess=$(
    $scripts/running/sbatch_bare.sh \
    $SLURM_ARGS_GENERIC \
    --dependency=afterok:$id_download \
    $SLURM_LOG_ARGS \
    $scripts/preprocessing/preprocess_generic.sh \
    $base $src $trg $model_name $dry_run $seed "$training_corpora" \
    $pose_type $sentencepiece_vocab_size $force_target_fps $normalize_poses "$testing_corpora"
)

echo "  id_preprocess: $id_preprocess | $logs_sub_sub/slurm-$id_preprocess.out" | tee -a $logs_sub_sub/MAIN

# Sockeye prepare (depends on preprocess)

id_prepare=$(
    $scripts/running/sbatch_bare.sh \
    $SLURM_ARGS_GENERIC \
    --dependency=afterok:$id_preprocess \
    $SLURM_LOG_ARGS \
    $scripts/preprocessing/prepare_generic.sh \
    $base $src $trg $model_name $seed $pose_type $bucket_scaling $max_seq_len_source $bucket_width
)

echo "  id_prepare: $id_prepare | $logs_sub_sub/slurm-$id_prepare.out"  | tee -a $logs_sub_sub/MAIN

# Sockeye train (depends on prepare)

id_train=$(
    $scripts/running/sbatch_bare.sh \
    $SLURM_ARGS_VOLTA_TRAIN \
    --dependency=afterok:$id_prepare \
    $SLURM_LOG_ARGS \
    $scripts/training/train_generic.sh \
    $base $src $trg $model_name $dry_run $seed $pose_type $bucket_scaling $max_seq_len_source \
    $bucket_width $initial_learning_rate $dropout $num_layers_encoder $num_layers_decoder
)

echo "  id_train: $id_train | $logs_sub_sub/slurm-$id_train.out"  | tee -a $logs_sub_sub/MAIN

# translate test set(s) (depends on train)

id_translate=$(
    $scripts/running/sbatch_bare.sh \
    $SLURM_ARGS_VOLTA_TRANSLATE \
    --dependency=afterany:$id_train \
    $SLURM_LOG_ARGS \
    $scripts/translation/translate_generic.sh \
    $base $src $trg $model_name $dry_run "$testing_corpora"
)

echo "  id_translate: $id_translate | $logs_sub_sub/slurm-$id_translate.out"  | tee -a $logs_sub_sub/MAIN

# evaluate BLEU and other metrics (depends on translate)

id_evaluate=$(
    $scripts/running/sbatch_bare.sh \
    $SLURM_ARGS_GENERIC \
    --dependency=afterok:$id_translate \
    $SLURM_LOG_ARGS \
    $scripts/evaluation/evaluate_generic.sh \
    $base $src $trg $model_name "$testing_corpora"
)

echo "  id_evaluate: $id_evaluate | $logs_sub_sub/slurm-$id_evaluate.out"  | tee -a $logs_sub_sub/MAIN
