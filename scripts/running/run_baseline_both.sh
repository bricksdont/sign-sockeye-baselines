#! /bin/bash

base=/net/cephfs/shares/volk.cl.uzh/mathmu/sign-sockeye-baselines
scripts=$base/scripts

local_download_data="/net/cephfs/shares/volk.cl.uzh/EASIER/WMT_Shared_Task"

# DSGS -> German

src="dsgs"
trg="de"

# baseline

model_name="baseline_both"

pose_type="openpose"

training_corpora="focusnews srf"
testing_corpora="test dev_unseen test_unseen"

force_target_fps="false"
normalize_poses="false"

. $scripts/running/run_generic.sh
