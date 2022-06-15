#! /bin/bash

base=/net/cephfs/shares/volk.cl.uzh/mathmu/sign-sockeye-baselines
scripts=$base/scripts

local_download_data="/net/cephfs/shares/volk.cl.uzh/EASIER/WMT_Shared_Task"

# DSGS -> German

src="dsgs"
trg="de"

# dry runs of all steps

dry_run="true"

# baseline

model_name="dry_run_baseline_both"

pose_type="openpose"

training_corpora="focusnews"
testing_corpora="test dev_unseen test_unseen"

force_target_fps="true"
normalize_poses="true"

bucket_scaling="true"

# this argument is for dry runs only, set to "true" to also repeat downloads (or linking)

repeat_download_step="false"

# delete files for this model to rerun everything

. $scripts/running/prompt_to_delete_dry_run_folders.sh

. $scripts/running/run_generic.sh
