#! /bin/bash

# calling process needs to set:
# base
# $src
# $trg
# $model_name
# $dry_run
# $testing_corpora

base=$1
src=$2
trg=$3
model_name=$4
dry_run=$5
testing_corpora=$6

venvs=$base/venvs
scripts=$base/scripts

eval "$(conda shell.bash hook)"
source activate $venvs/sockeye3

# after ativating the env on purpose

set -u

beam_size="5"
batch_size="64"
length_penalty_alpha="1.0"

data=$base/data
data_sub=$data/${src}-${trg}
data_sub_sub=$data_sub/$model_name

models=$base/models
models_sub=$models/${src}-${trg}
models_sub_sub=$models_sub/$model_name

translations=$base/translations
translations_sub=$translations/${src}-${trg}
translations_sub_sub=$translations_sub/$model_name

# fail with non-zero status if there is no model checkpoint,
# to signal to downstream dependencies that they cannot be satisfied

if [[ ! -e $models_sub_sub/params.best ]]; then
    echo "There is no single model checkpoint, file does not exist:"
    echo "$models_sub_sub/params.best"
    exit 1
fi

mkdir -p $translations_sub_sub

# beam translation

if [[ $dry_run == "true" ]]; then
    # redefine params
    beam_size=1
    batch_size=2
    dry_run_additional_args="--use-cpu"
else
    dry_run_additional_args=""
fi

for test_corpus in $testing_corpora; do

    input=$data_sub_sub/$test_corpus.h5
    output_pieces=$translations_sub_sub/$test_corpus.pieces.trg
    output=$translations_sub_sub/$test_corpus.trg

    if [[ -s $output ]]; then
      echo "Translations exist: $output"

      num_lines_input=$(python $scripts/translation/get_size_of_h5_dataset.py $input)
      num_lines_output=$(cat $output | wc -l)

      if [[ $num_lines_input == $num_lines_output ]]; then
          echo "output exists and number of lines are equal to input:"
          echo "$input == $output"
          echo "$num_lines_input == $num_lines_output"
          echo "Skipping."
          continue
      else
          echo "$input != $output"
          echo "$num_lines_input != $num_lines_output"
          echo "Repeating step."
      fi
    fi

    # 1-best translation with beam

    OMP_NUM_THREADS=1 python -m sockeye.translate \
            -i $input \
            -o $output_pieces \
            -m $models_sub_sub \
            --beam-size $beam_size \
            --length-penalty-alpha $length_penalty_alpha \
            --device-id 0 \
            --batch-size $batch_size $dry_run_additional_args

    # undo pieces

    cat $output_pieces | sed 's/ //g;s/▁/ /g' > $output

done
