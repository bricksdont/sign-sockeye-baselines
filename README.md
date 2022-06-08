# sign-sockeye-baselines

## Attribution required

If you use any of our code, please cite this repository as follows:

    @misc{mueller2022easier-gloss-translation-models,
        title={Sockeye baseline models for sign language translation},
        author={M\"{u}ller, Mathias and Rios, Annette and Moryossef, Amit},
        howpublished={\url{https://github.com/bricksdont/sign-sockeye-baselines}},
        year={2022}
    }

## Making changes to this repository

The best way of collaborating on this code is to create new branches and then create pull request based on changes in
a branch.

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

Finally, set the variable `local_download_data` to indicate where you downloaded our training
data. The path must be the folder which contains the two sub-folders `srf` and `focusnews`.

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

## Ideas for improving this baseline

- find bugs :-)
- use both SRF and focusnews training data in the same system
- also use monolingual subtitles (around 2000 files)
- openpose do not take the first person's keypoints, be more elaborate
- different / better normalization or scaling of poses
- augmentation, frame dropout from pose_format
- graph encoder for poses
- add mediapipe processing
