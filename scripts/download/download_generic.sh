#! /bin/bash

set -u

# calling process needs to set:
# $base
# $training_corpora
# $local_download_data

base=$1
training_corpora=$2
local_download_data=$3

scripts=$base/scripts
download=$base/download
venvs=$base/venvs

mkdir -p $download

# path to local download of training data
# (could be replaced with a download once our data is available online)

original_data=$local_download_data

# also link our unseen dev and test data (local links, participants will only have partial access to this at first)

all_corpora="$training_corpora dev test"

for corpus in $all_corpora; do

    download_sub=$download/$corpus

    if [[ -d $download_sub ]]; then
          echo "download_sub already exists: $download_sub"
          echo "Skipping. Delete files to repeat step."
          continue
    fi

    mkdir -p $download_sub

    # stand-in for actual download from an online source: link to local files
    # TODO: perhaps change once our data is online

    if [[ $corpus == "srf" ]]; then
        original_data_sub=$original_data/$corpus/parallel
    else
        original_data_sub=$original_data/$corpus
    fi

    for sub_folder in subtitles openpose mediapipe videos; do
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
