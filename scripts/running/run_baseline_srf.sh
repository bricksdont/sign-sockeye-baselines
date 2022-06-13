#! /bin/bash

base=/net/cephfs/shares/volk.cl.uzh/mathmu/sign-sockeye-baselines
scripts=$base/scripts

local_download_data="/net/cephfs/shares/volk.cl.uzh/EASIER/WMT_Shared_Task"

# DSGS -> German

src="dsgs"
trg="de"

# baseline

model_name="baseline_srf"

pose_type="openpose"

training_corpora="srf"
testing_corpora="test unseen"

force_target_fps="false"
normalize_poses="true"
bucket_scaling="false"

. $scripts/running/run_generic.sh
