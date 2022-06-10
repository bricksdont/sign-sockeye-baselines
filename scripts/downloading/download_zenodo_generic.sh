# calling process needs to set:
# $scripts
# $download_sub_zenodo
# $training_corpus
# $zenodo_deposit_id
# $zenodo_token

if [[ $zenodo_token == "none" ]]; then
    echo "Cannot download data without token. Set environment variable: 'ZENODO_TOKEN_[name of token]'"
    exit 1
fi

# using private link to set cookies

curl --cookie-jar $download_sub_zenodo/zenodo-cookies.txt \
    "https://zenodo.org/record/${zenodo_deposit_id}?token=${zenodo_token}"

# query the API for information about the deposit, extract the data link from the response

curl --cookie $download_sub_zenodo/zenodo-cookies.txt \
    "https://zenodo.org/api/records/${zenodo_deposit_id}" > $download_sub_zenodo/api_response.json

zip_link=$(python3 $scripts/downloading/get_zip_link_from_json.py --input $download_sub_zenodo/api_response.json)

echo "Zenodo link found: $zip_link"

curl --cookie $download_sub_zenodo/zenodo-cookies.txt $zip_link > $download_sub_zenodo/$training_corpus.zip

(cd $download_sub_zenodo && unzip $training_corpus.zip)

# move everything to enclosing folder

mv $download_sub_zenodo/$training_corpus/* $download_sub_zenodo/

rm $download_sub_zenodo/$training_corpus
rm $download_sub_zenodo/$training_corpus.zip
