#! /bin/bash

set -u

# calling process needs to set:
# $base
# $training_corpora

base=$1
training_corpora=$2

scripts=$base/scripts
download=$base/download
venvs=$base/venvs

mkdir -p $download

# TODO: change once our data is online

original_data=/net/cephfs/shares/volk.cl.uzh/EASIER/WMT_Shared_Task/

for training_corpus in $training_corpora; do

    download_sub=$download/$training_corpus

    if [[ -d $download_sub ]]; then
          echo "download_sub already exists: $download_sub"
          echo "Skipping. Delete files to repeat step."
          continue
    fi

    mkdir -p $download_sub

    # stand-in for actual download from an online source: link to local files
    # TODO: change once our data is online

    if [[ $training_corpus == "srf" ]]; then
        original_data_sub=$original_data/$training_corpus/parallel
    else
        original_data_sub=$original_data/$training_corpus
    fi

    for sub_folder in subtitles openpose mediapipe; do
        original_data_sub_sub=$original_data_sub/$sub_folder
        download_sub_sub=$download_sub/$sub_folder

        mkdir -p $download_sub_sub

        for original_file in $original_data_sub_sub/*; do
            original_basename=$(basename $original_file)
            ln -s $original_file $download_sub_sub/$original_basename
        done
    done

done

echo "Sizes of files:"

ls -lh $download/*/*/*
