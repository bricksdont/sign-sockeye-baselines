#! /bin/bash

set -u

# calling process needs to set:
# $base
# $src
# $trg
# $model_name
# $dry_run
# $seed
# $training_corpus
# $fps
# $pose_type


base=$1
src=$2
trg=$3
model_name=$4
dry_run=$5
seed=$6
training_corpus=$7
fps=$8
pose_type=$9

download=$base/download
data=$base/data
venvs=$base/venvs
scripts=$base/scripts
shared_models=$base/shared_models

mkdir -p $shared_models

# subfolders

download_sub=$download/$training_corpus
data_sub=$data/${src}-${trg}
shared_models_sub=$shared_models/${src}-${trg}

# overwrite subfolder names to make it easier to read

data_sub=$data_sub/$model_name
shared_models_sub=$shared_models_sub/$model_name

mkdir -p $shared_models_sub

eval "$(conda shell.bash hook)"
source activate $venvs/sockeye3

MOSES=$base/tools/moses-scripts/scripts
TOKENIZER=$MOSES/tokenizer

DEVTEST_SIZE=100

DRY_RUN_TRAIN_SIZE=10000
DRY_RUN_DEVTEST_SIZE=2

SENTENCEPIECE_VOCAB_SIZE=1000

CORPORA_EXCEPT_TRAIN="dev test"
ALL_CORPORA="$CORPORA_EXCEPT_TRAIN train"

echo "data_sub: $data_sub"

# measure time

SECONDS=0

#################

if [[ -d $data_sub ]]; then
    echo "Folder already exists: $data_sub"
    echo "Skipping. Delete files to repeat step."
    exit 0
fi

mkdir -p $data_sub

# truncate all data if dry run

if [[ $dry_run == "true" ]]; then
    train_size=$DRY_RUN_TRAIN_SIZE
    devtest_size=$DRY_RUN_DEVTEST_SIZE
else
    train_size="-1"
    devtest_size=$DEVTEST_SIZE
fi

# convert downloaded data to text and h5 format, and create train/dev/test split

python $scripts/preprocessing/convert_and_split_data.py \
    --download-sub $download_sub \
    --data-sub $data_sub \
    --seed $seed \
    --train-size $train_size \
    --devtest-size $devtest_size \
    --fps $fps \
    --pose-type $pose_type

# prenormalization for train data

for corpus in $all_corpora; do
      cat $data_sub/$corpus.trg | \
      perl -CS -pe 'tr[\x{9}\x{A}\x{D}\x{20}-\x{D7FF}\x{E000}-\x{FFFD}\x{10000}-\x{10FFFF}][]cd;' | \
      perl -CS -pe 's/\&\s*\#\s*160\s*\;/ /g' \
      > $data_sub/$corpus.prenorm.trg
done

# normalize train data

cat $data_sub/train.prenorm.trg | \
    ${TOKENIZER}/replace-unicode-punctuation.perl | \
    ${TOKENIZER}/remove-non-printing-char.perl | \
    ${TOKENIZER}/deescape-special-chars.perl | \
    sed 's/  */ /g;s/^ *//g;s/ *$//g' > \
        $data_sub/train.normalized.trg

# normalize dev / test data + other test corpora

for corpus in $corpora_except_train; do
    cat $data_sub/$corpus.prenorm.trg | \
    ${TOKENIZER}/replace-unicode-punctuation.perl | \
    ${TOKENIZER}/remove-non-printing-char.perl | \
    ${TOKENIZER}/deescape-special-chars.perl | \
    sed 's/  */ /g;s/^ *//g;s/ *$//g' > \
        $data_sub/$corpus.normalized.trg
done

# remove sentences from dev if source or target is empty
# (otherwise leads to potential Sockeye error)

mv $data_sub/dev.$pose_type.h5 $data_sub/dev.before_remove_empty.h5
mv $data_sub/dev.normalized.trg $data_sub/dev.before_remove_empty.trg

python $scripts/preprocessing/remove_if_source_or_target_empty.py \
    --input-src $data_sub/dev.before_remove_empty.h5 \
    --input-trg $data_sub/dev.before_remove_empty.trg \
    --output-src $data_sub/dev.h5 \
    --output-trg $data_sub/dev.normalized.trg


echo "sentencepiece_vocab_size=$SENTENCEPIECE_VOCAB_SIZE"

# learn sentencepiece model on train target

# determine character coverage

num_characters=$(head -n 1000000 $data_sub/train.normalized.trg | python $scripts/num_chars.py | wc -l)

if [[ $num_characters -gt 1000 ]]; then
    character_coverage=0.9995
else
    character_coverage=1.0
fi

python $scripts/preprocessing/train_sentencepiece.py \
  --model-prefix $shared_models_sub/trg.sentencepiece \
  --input $data_sub/train.normalized.trg \
  --vocab-size $SENTENCEPIECE_VOCAB_SIZE \
  --character-coverage $character_coverage \
  --input-sentence-size=$SENTENCEPIECE_MAX_LINES

# apply SP model to train, test and dev

for corpus in $all_corpora; do
    cat $data_sub/$corpus.normalized.trg | \
        python $scripts/preprocessing/apply_sentencepiece.py \
            --model $shared_models_sub/$lang.sentencepiece.model \
                > $data_sub/$corpus.pieces.trg
done

# sizes
echo "Sizes of all files:"

wc -l $data_sub/*
wc -l $shared_models_sub/*

echo "time taken:"
echo "$SECONDS seconds"
