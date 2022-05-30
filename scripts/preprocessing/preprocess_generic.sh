#! /bin/bash

# calling process needs to set:
# $base
# $src
# $trg
# $model_name
# $dry_run
# $seed
# $training_corpora
# $fps
# $pose_type
# $sentencepiece_vocab_size


base=$1
src=$2
trg=$3
model_name=$4
dry_run=$5
seed=$6
training_corpora=$7
fps=$8
pose_type=$9
sentencepiece_vocab_size=${10}

download=$base/download
data=$base/data
venvs=$base/venvs
scripts=$base/scripts
shared_models=$base/shared_models

mkdir -p $shared_models

# subfolders

data_sub=$data/${src}-${trg}
shared_models_sub=$shared_models/${src}-${trg}

# overwrite subfolder names to make it easier to read

data_sub=$data_sub/$model_name
shared_models_sub=$shared_models_sub/$model_name

mkdir -p $shared_models_sub

eval "$(conda shell.bash hook)"
source activate $venvs/sockeye3

# after ativating the env on purpose

set -u

MOSES=$base/tools/moses-scripts/scripts
TOKENIZER=$MOSES/tokenizer

DEVTEST_SIZE=100

DRY_RUN_TRAIN_SIZE=100
DRY_RUN_DEVTEST_SIZE=2

SENTENCEPIECE_VOCAB_SIZE=1000

SENTENCEPIECE_MAX_LINES=1000000

SUBSETS_EXCEPT_TRAIN="dev test"
ALL_SUBSETS="$SUBSETS_EXCEPT_TRAIN train"

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
    train_size_arg="--train-size $DRY_RUN_TRAIN_SIZE"
    devtest_size=$DRY_RUN_DEVTEST_SIZE
    dry_run_arg="--dry-run"
else
    train_size_arg=""
    devtest_size=$DEVTEST_SIZE
    dry_run_arg=""
fi

for training_corpus in $training_corpora; do

    download_sub=$download/$training_corpus

    # convert downloaded data to text and h5 format, and create train/dev/test split

    # --output-prefix naming logic: [prefix].{dev,test,train}.[for h5: openpose or mediapipe].{txt,h5}.

    python $scripts/preprocessing/convert_and_split_data.py \
        --download-sub $download_sub \
        --output-dir $data_sub \
        --output-prefix $training_corpus \
        --seed $seed \
        --devtest-size $devtest_size \
        --fps $fps \
        --pose-type $pose_type $train_size_arg $dry_run_arg

done

# combine training corpora (poses and text separately)

for subset in $ALL_SUBSETS; do

    touch $data_sub/$subset.txt

    # combine texts

    for training_corpus in $training_corpora; do

        # training corpora: focusnews, srf

        cat $data_sub/$training_corpus.$subset.txt >> $data_sub/$subset.txt
    done

    # combine pose files

    python $scripts/preprocessing/combine_h5_datasets.py \
        --inputs $data_sub/*.$subset.h5 \
        --output $data_sub/$subset.h5

done

# prenormalization for all subsets (targets only from here on)

for subset in $ALL_SUBSETS; do
      cat $data_sub/$subset.txt | \
      perl -CS -pe 'tr[\x{9}\x{A}\x{D}\x{20}-\x{D7FF}\x{E000}-\x{FFFD}\x{10000}-\x{10FFFF}][]cd;' | \
      perl -CS -pe 's/\&\s*\#\s*160\s*\;/ /g' \
      > $data_sub/$subset.prenorm.trg
done

# normalize train data

cat $data_sub/train.prenorm.trg | \
    ${TOKENIZER}/replace-unicode-punctuation.perl | \
    ${TOKENIZER}/remove-non-printing-char.perl | \
    ${TOKENIZER}/deescape-special-chars.perl | \
    sed 's/  */ /g;s/^ *//g;s/ *$//g' > \
        $data_sub/train.normalized.trg

# normalize dev / test data + other test corpora

for subset in $SUBSETS_EXCEPT_TRAIN; do
    cat $data_sub/$subset.prenorm.trg | \
    ${TOKENIZER}/replace-unicode-punctuation.perl | \
    ${TOKENIZER}/remove-non-printing-char.perl | \
    ${TOKENIZER}/deescape-special-chars.perl | \
    sed 's/  */ /g;s/^ *//g;s/ *$//g' > \
        $data_sub/$subset.normalized.trg
done

echo "sentencepiece_vocab_size=$sentencepiece_vocab_size"

# learn sentencepiece model on train target

# determine character coverage

num_characters=$(head -n 1000000 $data_sub/train.normalized.trg | python $scripts/preprocessing/num_chars.py | wc -l)

if [[ $num_characters -gt 1000 ]]; then
    character_coverage=0.9995
else
    character_coverage=1.0
fi

python $scripts/preprocessing/train_sentencepiece.py \
  --model-prefix $shared_models_sub/trg.sentencepiece \
  --input $data_sub/train.normalized.trg \
  --vocab-size $sentencepiece_vocab_size \
  --character-coverage $character_coverage \
  --input-sentence-size=$SENTENCEPIECE_MAX_LINES

# apply SP model to train, test and dev

for subset in $ALL_SUBSETS; do
    cat $data_sub/$subset.normalized.trg | \
        python $scripts/preprocessing/apply_sentencepiece.py \
            --model $shared_models_sub/trg.sentencepiece.model \
                > $data_sub/$subset.pieces.trg
done

# sizes
echo "Sizes of all files:"

ls -lh $data_sub/*.h5
wc -l $data_sub/*.txt
wc -l $data_sub/*.trg
wc -l $shared_models_sub/*

echo "time taken:"
echo "$SECONDS seconds"
