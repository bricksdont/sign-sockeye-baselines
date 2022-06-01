#! /bin/bash

base=/net/cephfs/shares/volk.cl.uzh/mathmu/sign-sockeye-baselines
scripts=$base/scripts

# DSGS -> German

src="dsgs"
trg="de"

# dry runs of all steps

dry_run="true"

# baseline

model_name="dry_run_srf"

pose_type="openpose"

training_corpora="srf"
testing_corpora="test unseen"

force_target_fps="false"
normalize_poses="false"

# delete files for this model to rerun everything

sub_folders="data shared_models prepared models translations evaluations"

echo "Could delete the following folders related to $src-$trg/$model_name:"

for sub_folder in $sub_folders; do
  echo "$base/$sub_folder/$src-$trg/$model_name"
done

read -p "Delete? (y/n) " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    for sub_folder in $sub_folders; do
      rm -rf $base/$sub_folder/$src-$trg/$model_name
    done

    if [[ $repeat_download_step == "true" ]]; then
      for source in $training_corpora; do
          rm -rf "$base/data/download/$source"
        done
    fi
fi

. $scripts/running/run_generic.sh
