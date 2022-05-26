# sign-sockeye-baselines

## Basic setup

Create a venv:

    ./scripts/setup/create_venv.sh

Then install required software:

    ./scripts/setup/install.sh

## Dry run

Try to create all files and run all scripts, but on CPU only and exit immediately without any actual computation:

    ./scripts/running/dry_run_baseline.sh

## Run a bilingual baseline

Train a baseline system for DSGS -> DE:

    ./scripts/running/run_baseline.sh
