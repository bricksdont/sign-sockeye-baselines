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
# $fps
# $pose_type
# $sentencepiece_vocab_size
# $force_target_fps
# $normalize_poses
# $bucket_scaling
# $local_download_data
#
# optional environment variables to be set when calling a run script (these are private tokens that should not appear
# in logs or commits):
# $ZENODO_TOKEN_FOCUSNEWS
# $ZENODO_TOKEN_SRF_POSES
# $ZENODO_TOKEN_SRF_VIDEOS_SUBTITLES

scripts=$base/scripts
logs=$base/logs

source activate $base/venvs/sockeye3

logs_sub=$logs/${src}-${trg}
logs_sub_sub=$logs_sub/$model_name

mkdir -p $logs_sub_sub

# if variables are undefined, set to avoid confusion

if [ -z "$dry_run" ]; then
    dry_run="false"
fi

if [ -z "$local_download_data" ]; then
    local_download_data="/net/cephfs/shares/volk.cl.uzh/EASIER/WMT_Shared_Task/"
fi

if [ -z "$training_corpora" ]; then
    training_corpora="focusnews"
fi

if [ -z "$testing_corpora" ]; then
    testing_corpora="test"
fi

if [ -z "$fps" ]; then
    fps="25"
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
    normalize_poses="false"
fi

if [ -z "$bucket_scaling" ]; then
    bucket_scaling="false"
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
echo "DRY RUN: $dry_run" | tee -a $logs_sub_sub/MAIN

# download corpora

id_download=$(python  -c 'import uuid; print(uuid.uuid4().hex)')

$scripts/downloading/download_generic.sh \
    $base "$training_corpora" $local_download_data "$testing_corpora" \
    $zenodo_token_focusnews $zenodo_token_srf_poses $zenodo_token_srf_videos_subtitles \
    > $logs_sub_sub/slurm-$id_download.out 2> $logs_sub_sub/slurm-$id_download.out

echo "  id_download: $id_download | $logs_sub_sub/slurm-$id_download.out" | tee -a $logs_sub_sub/MAIN

# preprocess: Combine datasets, hold out data, normalize, SPM (depends on download)

id_preprocess=$(python  -c 'import uuid; print(uuid.uuid4().hex)')

$scripts/preprocessing/preprocess_generic.sh \
    $base $src $trg $model_name $dry_run $seed "$training_corpora" \
    $fps $pose_type $sentencepiece_vocab_size $force_target_fps $normalize_poses "$testing_corpora" \
    > $logs_sub_sub/slurm-$id_preprocess.out 2> $logs_sub_sub/slurm-$id_preprocess.out

echo "  id_preprocess: $id_preprocess | $logs_sub_sub/slurm-$id_preprocess.out" | tee -a $logs_sub_sub/MAIN

# Sockeye prepare (depends on preprocess)

id_prepare=$(python  -c 'import uuid; print(uuid.uuid4().hex)')

$scripts/preprocessing/prepare_generic.sh \
    $base $src $trg $model_name $seed $pose_type $bucket_scaling \
    > $logs_sub_sub/slurm-$id_prepare.out 2> $logs_sub_sub/slurm-$id_prepare.out

echo "  id_prepare: $id_prepare | $logs_sub_sub/slurm-$id_prepare.out"  | tee -a $logs_sub_sub/MAIN

# Sockeye train (depends on prepare)

id_train=$(python  -c 'import uuid; print(uuid.uuid4().hex)')

$scripts/training/train_generic.sh \
    $base $src $trg $model_name $dry_run $seed $pose_type $bucket_scaling \
    > $logs_sub_sub/slurm-$id_train.out 2> $logs_sub_sub/slurm-$id_train.out

echo "  id_train: $id_train | $logs_sub_sub/slurm-$id_train.out"  | tee -a $logs_sub_sub/MAIN

# translate test set(s) (depends on train)

id_translate=$(python  -c 'import uuid; print(uuid.uuid4().hex)')

$scripts/translation/translate_generic.sh \
    $base $src $trg $model_name $dry_run "$testing_corpora" \
    > $logs_sub_sub/slurm-$id_translate.out 2> $logs_sub_sub/slurm-$id_translate.out

echo "  id_translate: $id_translate | $logs_sub_sub/slurm-$id_translate.out"  | tee -a $logs_sub_sub/MAIN

# evaluate BLEU and other metrics (depends on translate)

id_evaluate=$(python  -c 'import uuid; print(uuid.uuid4().hex)')

$scripts/evaluation/evaluate_generic.sh \
    $base $src $trg $model_name "$testing_corpora" \
    > $logs_sub_sub/slurm-$id_evaluate.out 2> $logs_sub_sub/slurm-$id_evaluate.out

echo "  id_evaluate: $id_evaluate | $logs_sub_sub/slurm-$id_evaluate.out"  | tee -a $logs_sub_sub/MAIN
