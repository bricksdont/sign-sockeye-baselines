#! /bin/bash

base=/net/cephfs/shares/volk.cl.uzh/mathmu/sign-sockeye-baselines
scripts=$base/scripts

# DSGS -> German

src="dsgs"
trg="de"

# baseline

model_name="baseline_focusnews"

pose_type="openpose"

training_corpora="focusnews"
testing_corpora="test unseen"

. $scripts/running/run_generic.sh
