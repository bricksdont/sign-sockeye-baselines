#! /bin/bash

set -u

# calling process needs to set:
# $base
# $training_corpora
# $testing_corpora
# $local_download_data
# $zenodo_token_focusnews
# $zenodo_token_srf_poses
# $zenodo_token_srf_videos_subtitles

base=$1
training_corpora=$2
local_download_data=$3
testing_corpora=$4
zenodo_token_focusnews=$5
zenodo_token_srf_poses=$6
zenodo_token_srf_videos_subtitles=$7

scripts=$base/scripts
download=$base/download
venvs=$base/venvs

mkdir -p $download

# these IDs change if a new version is released on Zenodo

ZENODO_DEPOSIT_ID_FOCUSNEWS=6631159
ZENODO_DEPOSIT_ID_SRF_POSES=6631275
ZENODO_DEPOSIT_ID_SRF_VIDEOS_SUBTITLES=6637392

# only download if user indicated they have *not* already downloaded elsewhere

for training_corpus in $training_corpora; do

    download_sub=$download/$training_corpus

    if [[ -d $download_sub ]]; then
          echo "download_sub already exists: $download_sub"
          echo "Skipping. Delete files to repeat step."
          continue
    fi

    mkdir -p $download_sub

    if [[ $local_download_data == "false" ]]; then

        # then download the data from Zenodo

        if [[ $training_corpus == "focusnews" ]]; then

            download_sub_zenodo=$download_sub
            zenodo_deposit_id=$ZENODO_DEPOSIT_ID_FOCUSNEWS
            zenodo_token=$zenodo_token_focusnews

            . $scripts/downloading/download_zenodo_generic.sh
        else
            # assume training corpus is SRF

            # download poses

            download_sub_zenodo=$download_sub/zenodo_poses
            zenodo_deposit_id=$ZENODO_DEPOSIT_ID_SRF_POSES
            zenodo_token=$zenodo_token_srf_poses

            mkdir -p $download_sub_zenodo

            . $scripts/downloading/download_zenodo_generic.sh

            # download videos and subtitles

            download_sub_zenodo=$download_sub/zenodo_videos_subtitles
            zenodo_deposit_id=$ZENODO_DEPOSIT_ID_SRF_VIDEOS_SUBTITLES
            zenodo_token=$zenodo_token_srf_videos_subtitles

            mkdir -p $download_sub_zenodo

            . $scripts/downloading/download_zenodo_generic.sh

            # finally combine data from both folders srf/zenodo_poses and srf/zenodo_videos_subtitles back into one

            # note: this ignores the monolingual subtitles as they are not used by the baseline systems

            ln -s  $download_sub/zenodo_videos_subtitles/parallel/videos $download_sub/videos
            ln -s  $download_sub/zenodo_videos_subtitles/parallel/subtitles $download_sub/subtitles

            ln -s  $download_sub/zenodo_poses/parallel/openpose $download_sub/openpose
            ln -s  $download_sub/zenodo_poses/parallel/mediapipe $download_sub/mediapipe

        fi

    else
        # in that case link existing files

        corpus_name=$training_corpus
        original_corpus_name=$training_corpus

        . $scripts/downloading/download_link_folder_generic.sh
    fi
done

# download or link dev and test data if requested

for testing_corpus in $testing_corpora; do

    # do nothing if the test corpus is "dev" or "test" since these are
    # slices of the training data downloaded (or linked) above

    if [[ $testing_corpus == "dev" || $testing_corpus == "test" ]]; then
        continue
    fi

    if [[ $testing_corpus == "dev_unseen" ]]; then
        original_corpus_name="dev"
    elif [[ $testing_corpus == "test_unseen" ]]; then
        # assume test
        original_corpus_name="test"
    fi

    download_sub=$download/$testing_corpus

    if [[ -d $download_sub ]]; then
          echo "download_sub already exists: $download_sub"
          echo "Skipping. Delete files to repeat step."
          continue
    fi

    mkdir -p $download_sub

    if [[ $local_download_data == "false" ]]; then

        # TODO: download dev and test data instead of also linking locally here once it is available online
        local_download_data="/net/cephfs/shares/volk.cl.uzh/EASIER/WMT_Shared_Task"

        corpus_name=$testing_corpus

        . $scripts/downloading/download_link_folder_generic.sh

    else
        # in that case link existing files

        corpus_name=$testing_corpus

        . $scripts/downloading/download_link_folder_generic.sh
    fi
done

echo "Sizes of files:"

ls -lh $download/*/*/*
