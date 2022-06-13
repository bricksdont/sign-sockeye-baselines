# calling script needs to set:
# $base
# $src
# $trg
# $model_name
# $training_corpora
# $testing_corpora
# $repeat_download_step

set -u

echo "REPEAT_DOWNLOAD_STEP: $repeat_download_step"

sub_folders="data shared_models prepared models translations evaluations"

echo "Could delete the following folders related to $src-$trg/$model_name:"

for sub_folder in $sub_folders; do
  echo "$base/$sub_folder/$src-$trg/$model_name"
done

if [[ $repeat_download_step == "true" ]]; then
    for training_corpus in $training_corpora; do
        echo "$base/download/$training_corpus"
    done

    for testing_corpus in $testing_corpora; do

        if [[ $testing_corpus == "dev" || $testing_corpus == "test" ]]; then
            continue
        fi
        echo "$base/download/$testing_corpus"
    done
fi

read -p "Delete? (y/n) " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    for sub_folder in $sub_folders; do
        rm -rf $base/$sub_folder/$src-$trg/$model_name
    done

    if [[ $repeat_download_step == "true" ]]; then
        for training_corpus in $training_corpora; do
            rm -rf "$base/download/$training_corpus"
        done

        for testing_corpus in $testing_corpora; do

            if [[ $testing_corpus == "dev" || $testing_corpus == "test" ]]; then
                continue
            fi
            rm -rf "$base/download/$testing_corpus"
        done
    fi
fi

set +u
