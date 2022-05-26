#! /bin/bash

set -u

# calling process needs to set:
# $base
# $src
# $trg
# $model_name
# $dry_run
# $seed

base=$1
src=$2
trg=$3
model_name=$4
dry_run=$5
seed=$6

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

MOSES=$base/tools/moses-scripts/scripts
TOKENIZER=$MOSES/tokenizer

DRY_RUN_TRAIN_SIZE=14000
DRY_RUN_DEVTEST_SIZE=2

SENTENCEPIECE_VOCAB_SIZE=1000

CORPORA_EXCEPT_TRAIN="dev test"
ALL_CORPORA="$CORPORA_EXCEPT_TRAIN train"

echo "data_sub: $data_sub"

# measure time

SECONDS=0

#################

if [[ -f $data_sub/test.pieces.src ]]; then
    echo "File already exists: $data_sub/test.pieces.src"
    echo "Skipping. Delete files to repeat step."
    exit 0
fi

mkdir -p $data_sub

source $language_pairs_script

echo "language_pairs: "
echo "${language_pairs[@]}"

for pair in "${language_pairs[@]}"; do

    # if particular files appear several times in the array, processing happens twice which is negligible

    pair=($pair)

    source=${pair[0]}
    src=${pair[1]}
    trg=${pair[2]}

      echo "Found (source, src, trg): ($source, $src, $trg)"

    download_sub=$data/download/$source

    for lang in $src $trg; do

        # extract data from download jsons

        for corpus in $ALL_CORPORA; do

            python $scripts/preprocessing/extract_key_from_json.py \
                --input-file $download_sub/$corpus.json \
                --output-file $data_sub/$source.$corpus.$lang \
                --key $lang
        done

        # truncate all files if this is a dry run

        if [[ $dry_run == "true" ]]; then

            for corpus in $CORPORA_EXCEPT_TRAIN; do
                mv $data_sub/$source.$corpus.$lang $data_sub/$source.$corpus.$lang.big
                head -n $DRY_RUN_DEVTEST_SIZE $data_sub/$source.$corpus.$lang.big > $data_sub/$source.$corpus.$lang
            done

            mv $data_sub/$source.train.$lang $data_sub/$source.train.$lang.big
            head -n $DRY_RUN_TRAIN_SIZE $data_sub/$source.train.$lang.big > $data_sub/$source.train.$lang
        fi

        # if lang is a gloss suffix, possibly lowercase or other preprocessing
        # if lang is spoken suffix, this step does nothing

        for corpus in $ALL_CORPORA; do
            python $scripts/preprocessing/preprocess_glosses.py \
                --input-file $data_sub/$source.$corpus.$lang \
                --output-file $data_sub/$source.$corpus.preprocessed.$lang \
                --lang $lang \
                --lowercase-glosses $lowercase_glosses \
                --generalize-dgs-glosses $generalize_dgs_glosses
        done

        # prenormalization for all corpora

        for corpus in $ALL_CORPORA; do
            cat $data_sub/$source.$corpus.preprocessed.$lang | \
            perl -CS -pe 'tr[\x{9}\x{A}\x{D}\x{20}-\x{D7FF}\x{E000}-\x{FFFD}\x{10000}-\x{10FFFF}][]cd;' | \
            perl -CS -pe 's/\&\s*\#\s*160\s*\;/ /g' \
            > $data_sub/$source.$corpus.prenorm.$lang
        done

        # normalize all corpora

        for corpus in $ALL_CORPORA; do
            cat $data_sub/$source.$corpus.prenorm.$lang | \
            ${TOKENIZER}/replace-unicode-punctuation.perl | \
            ${TOKENIZER}/remove-non-printing-char.perl | \
            ${TOKENIZER}/deescape-special-chars.perl | \
            sed 's/  */ /g;s/^ *//g;s/ *$//g' > \
                $data_sub/$source.$corpus.normalized.$lang
        done
    done

done

# learn sentencepiece model(s) on train

if [[ $spm_strategy == "joint" || $spm_strategy == "spoken-only" ]]; then

    # then train one spm model overall

    if [[ $spm_strategy == "joint" ]]; then
        # use all normalized train files

        cat $data_sub/*.train.normalized.* > $data_sub/train.normalized.all
    else
        # $spm_strategy == "spoken-only"
        # use normalized train files for all spoken languages

        echo -n "" > $data_sub/train.normalized.all

        for source in $ALL_SOURCES; do
            for lang in $SPOKEN_SUFFIXES; do
                if [[ -f $data_sub/$source.train.normalized.$lang ]]; then
                  cat $data_sub/$source.train.normalized.$lang >> $data_sub/train.normalized.all
                fi
            done
        done
    fi

    input=$data_sub/train.normalized.all
    model_prefix=$shared_models_sub/sentencepiece

    . $scripts/preprocessing/train_sentencepiece_generic.sh

elif [[ $spm_strategy == "separate" ]]; then
    # one spm model for spoken, one for gloss suffixes

    echo -n "" > $data_sub/train.normalized.spoken
    echo -n "" > $data_sub/train.normalized.gloss

    for source in $ALL_SOURCES; do
        for lang in $SPOKEN_SUFFIXES; do
            if [[ -f $data_sub/$source.train.normalized.$lang ]]; then
              cat $data_sub/$source.train.normalized.$lang >> $data_sub/train.normalized.spoken
            fi
        done

        for lang in $GLOSS_SUFFIXES; do
            if [[ -f $data_sub/$source.train.normalized.$lang ]]; then
              cat $data_sub/$source.train.normalized.$lang >> $data_sub/train.normalized.gloss
            fi
        done
    done

    for suffix in spoken gloss; do

        input=$data_sub/train.normalized.$suffix
        model_prefix=$shared_models_sub/$suffix.sentencepiece

        . $scripts/preprocessing/train_sentencepiece_generic.sh

    done
else
    echo "ERROR: Unknown sentencepiece strategy: $spm_strategy"
    echo "Specify one of: 'joint', 'separate', 'spoken-only'"
    exit 1
fi

# apply SP models to train, test and dev

if [[ $spm_strategy == "joint" ]]; then

    for source in $ALL_SOURCES; do
        for corpus in $ALL_CORPORA; do
            for suffix in $ALL_SUFFIXES; do

                if [[ -f $data_sub/$source.$corpus.normalized.$suffix ]]; then
                    cat $data_sub/$source.$corpus.normalized.$suffix | \
                        python $scripts/preprocessing/apply_sentencepiece.py \
                            --model $shared_models_sub/sentencepiece.model \
                                > $data_sub/$source.$corpus.pieces.$suffix
                fi
            done
        done
    done

elif [[ $spm_strategy == "spoken-only" ]]; then

    for source in $ALL_SOURCES; do
        for corpus in $ALL_CORPORA; do
            for suffix in $SPOKEN_SUFFIXES; do

                if [[ -f $data_sub/$source.$corpus.normalized.$suffix ]]; then
                    cat $data_sub/$source.$corpus.normalized.$suffix | \
                        python $scripts/preprocessing/apply_sentencepiece.py \
                            --model $shared_models_sub/sentencepiece.model \
                                > $data_sub/$source.$corpus.pieces.$suffix
                fi
            done

            for suffix in $GLOSS_SUFFIXES; do

                if [[ -f $data_sub/$source.$corpus.normalized.$suffix ]]; then
                    # applying spm model is a no-op for gloss data in this case

                    cp $data_sub/$source.$corpus.normalized.$suffix $data_sub/$source.$corpus.pieces.$suffix
                fi
            done
        done
    done

else
    # $spm_strategy == "separate"
    for source in $ALL_SOURCES; do
        for corpus in $ALL_CORPORA; do
            for suffix in $SPOKEN_SUFFIXES; do

                if [[ -f $data_sub/$source.$corpus.normalized.$suffix ]]; then
                    cat $data_sub/$source.$corpus.normalized.$suffix | \
                        python $scripts/preprocessing/apply_sentencepiece.py \
                            --model $shared_models_sub/spoken.sentencepiece.model \
                                > $data_sub/$source.$corpus.pieces.$suffix
                fi
            done

            for suffix in $GLOSS_SUFFIXES; do

                if [[ -f $data_sub/$source.$corpus.normalized.$suffix ]]; then
                    cat $data_sub/$source.$corpus.normalized.$suffix | \
                        python $scripts/preprocessing/apply_sentencepiece.py \
                            --model $shared_models_sub/gloss.sentencepiece.model \
                                > $data_sub/$source.$corpus.pieces.$suffix
                fi
            done
        done
    done
fi

# put together training data and correctly assign ".src" and ".trg" suffixes

for corpus in $ALL_CORPORA; do

    echo -n "" > $data_sub/$corpus.pieces.src
    echo -n "" > $data_sub/$corpus.pieces.trg

    for pair in "${language_pairs[@]}"; do
        pair=($pair)

        source=${pair[0]}
        src=${pair[1]}
        trg=${pair[2]}

        if [[ $multilingual == "true" ]]; then
             cat $data_sub/$source.$corpus.pieces.$src | \
                 python $scripts/preprocessing/add_tag_to_lines.py --tag "<2$trg>" \
                     > $data_sub/$source.$corpus.tag.$src

             cat $data_sub/$source.$corpus.tag.$src >> $data_sub/$corpus.pieces.src
        else
             cat $data_sub/$source.$corpus.pieces.$src >> $data_sub/$corpus.pieces.src
        fi
        cat $data_sub/$source.$corpus.pieces.$trg >> $data_sub/$corpus.pieces.trg
    done
done

# ratio etc filter for train

$MOSES/training/clean-corpus-n.perl -ignore-ratio $data_sub/train.pieces src trg $data_sub/train.clean 1 250

# remove sentences from dev if source or target is empty
# (otherwise leads to potential Sockeye error)

for corpus in dev; do
    for lang in src trg; do
        mv $data_sub/$corpus.pieces.$lang $data_sub/$corpus.pieces.before_remove_empty.$lang
    done

    python $scripts/preprocessing/remove_if_source_or_target_empty.py \
        --input-src $data_sub/$corpus.pieces.before_remove_empty.src \
        --input-trg $data_sub/$corpus.pieces.before_remove_empty.trg \
        --output-src $data_sub/$corpus.pieces.src \
        --output-trg $data_sub/$corpus.pieces.trg
done

# sizes
echo "Sizes of all files:"

wc -l $data_sub/*
wc -l $shared_models_sub/*

echo "time taken:"
echo "$SECONDS seconds"
