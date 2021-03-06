# calling process needs to set:
# $download_sub
# $corpus_name
# $original_corpus_name
# $local_download_data

if [[ $corpus_name == "srf" ]]; then
    local_download_data_sub=$local_download_data/$original_corpus_name/parallel
elif [[ $corpus_name == "test_unseen" ]]; then
    local_download_data_sub=$local_download_data/$original_corpus_name/dsgs-de
else
    local_download_data_sub=$local_download_data/$original_corpus_name
fi

for sub_folder in subtitles openpose mediapipe videos; do
    local_download_data_sub_sub=$local_download_data_sub/$sub_folder
    download_sub_sub=$download_sub/$sub_folder

    mkdir -p $download_sub_sub

    for original_file in $local_download_data_sub_sub/*; do
        original_basename=$(basename $original_file)
        linkname=$(python $scripts/preprocessing/get_linkname_unseen_corpus.py $original_basename)
        ln -s $original_file $download_sub_sub/$linkname
    done
done
