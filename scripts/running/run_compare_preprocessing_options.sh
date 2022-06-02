#! /bin/bash

base=/net/cephfs/shares/volk.cl.uzh/mathmu/sign-sockeye-baselines
scripts=$base/scripts

# DSGS -> German

src="dsgs"
trg="de"

# TOD: test mediapipe as well once available

pose_type="openpose"

testing_corpora="test unseen"

for training_corpus in srf focusnews; do
    for force_target_fps in false true; do
        for normalize_poses in false true; do

            model_name="training_corpus.$training_corpus+force_target_fps.$force_target_fps+normalize_poses.$normalize_poses"

            training_corpora=$training_corpus

            . $scripts/running/run_generic.sh
        done
    done
done
