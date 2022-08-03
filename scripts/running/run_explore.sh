#! /bin/bash

base=/net/cephfs/shares/volk.cl.uzh/mathmu/sign-sockeye-baselines
scripts=$base/scripts

local_download_data="/net/cephfs/shares/volk.cl.uzh/EASIER/WMT_Shared_Task"

# DSGS -> German

src="dsgs"
trg="de"

pose_type="mediapipe"

# known to be beneficial from previous experiments

bucket_scaling="true"
normalize_poses="true"
force_target_fps="true"

testing_corpora="test dev_unseen test_unseen"

seeds="2 3"
bucket_widths="8 16 32"
initial_learning_rates="0.001 0.0002 0.0001"
dropouts="0.1 0.2 0.5"
num_layers_encoders="6"
num_layers_decoders="6"

for seed in $seeds; do
  for training_corpus in srf focusnews both; do
      for bucket_width in $bucket_widths; do
          for initial_learning_rate in $initial_learning_rates; do
              for dropout in $dropouts; do
                  for num_layers_encoder in $num_layers_encoders; do
                      for num_layers_decoder in $num_layers_decoders; do

                      model_name="training_corpus.$training_corpus+pose_type.$pose_type+bucket_width.$bucket_width+initial_learning_rate.$initial_learning_rate+dropout.$dropout+num_layers_encoder.$num_layers_encoder+num_layers_decoder.$num_layers_decoder+seed.$seed"

                      if [[ $training_corpora == "both" ]]; then
                          training_corpora="srf focusnews"
                      else
                          training_corpora=$training_corpus
                      fi

                      . $scripts/running/run_generic.sh
                      done
                  done
              done
          done
      done
  done
done