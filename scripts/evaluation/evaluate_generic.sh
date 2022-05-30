#! /bin/bash

# calling process needs to set:
# base
# $src
# $trg
# $model_name
# $testing_corpora

base=$1
src=$2
trg=$3
model_name=$4
testing_corpora=$5

venvs=$base/venvs
scripts=$base/scripts

eval "$(conda shell.bash hook)"
source activate $venvs/sockeye3

# after ativating the env on purpose

set -u

data=$base/data
data_sub=$data/${src}-${trg}
data_sub_sub=$data_sub/$model_name

translations=$base/translations
translations_sub=$translations/${src}-${trg}
translations_sub_sub=$translations_sub/$model_name

evaluations=$base/evaluations
evaluations_sub=$evaluations/${src}-${trg}
evaluations_sub_sub=$evaluations_sub/$model_name

mkdir -p $evaluations_sub_sub

# compute case-sensitive BLEU and CHRF on detokenized data

chrf_beta=2

tokenize="true"

for corpus in $testing_corpora; do

    ref=$data_sub_sub/$corpus.trg
    hyp=$translations_sub_sub/$corpus.trg

    output_prefix=$evaluations_sub_sub/$corpus.trg

    output=$output_prefix.bleu

    . $scripts/evaluation/evaluate_bleu_more_generic.sh

    output=$output_prefix.chrf

    . $scripts/evaluation/evaluate_chrf_more_generic.sh

    done
