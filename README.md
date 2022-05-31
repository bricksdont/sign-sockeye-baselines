# sign-sockeye-baselines

## Type of machine required

A recent version of Ubuntu with conda installed. The code also
assumes a GPU is available on the system (a single V100 in our
baseline experiments).

## What to modify before running scripts

Top-level scripts (in `scripts/running`) assume that tasks should be submitted to SLURM
scheduling system. If you don't have SLURM, you need to replace calls to the script
`scripts/running/run_generic.sh` with `scripts/running/run_generic_no_slurm.sh`.

Another important variable to change is `base`, which is defined
in every top-level script (in `scripts/running`). `base` determines where
files and folders should be written.

## Basic setup

Create a venv:

    ./scripts/setup/create_venv.sh

Then install required software:

    ./scripts/setup/install.sh

## Dry run

Try to create all files and run all scripts, but on CPU only and exit immediately without any actual computation:

    ./scripts/running/dry_run_baseline_focusnews.sh

And a dry run for srf training data:

    ./scripts/running/dry_run_baseline_srf.sh

## Run a bilingual baseline

Train a baseline system for DSGS -> DE using focusnews data:

    scripts/running/run_baseline_focusnews.sh

A baseline using SRF data:

    scripts/running/run_baseline_srf.sh
