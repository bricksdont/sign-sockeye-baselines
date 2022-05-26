#! /bin/bash

base=/net/cephfs/shares/volk.cl.uzh/mathmu/sign-sockeye-baselines
scripts=$base/scripts

# DSGS -> German

src="dsgs"
trg="de"

# baseline

model_name="baseline"

training_corpora="focusnews"
testing_corpora="test"

. $scripts/running/run_generic.sh
