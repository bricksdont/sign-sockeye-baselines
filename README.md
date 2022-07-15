# sign-sockeye-baselines

## Attribution required

If you use any of our code, please cite this repository as follows:

    @misc{mueller2022sign-sockeye-baselines,
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
baseline experiments). CUDA and CuDNN must be installed.

## How to use this repository

If you would like to train baselines without modifying the code, execute one of the scripts in `scripts/running`. Each
of these scripts defines a "run". Our definition of a "run" is a complete experiment, in the sense that the script will
execute all steps necessary including downloading the data, preprocessing, training, translating and evaluating the system.

All steps have separate logs that can be inspected and if a step fails, the pipeline can be re-run without executing previous,
successful steps again. On a scheduling system like SLURM, steps will be submitted to the scheduler at the same time and
have a dependency on each other (e.g. `--after-ok`).

### What to modify before running scripts

Top-level scripts (in `scripts/running`) assume that tasks should be submitted to SLURM
scheduling system. If you don't have SLURM, you need to replace calls to the script
`scripts/running/run_generic.sh` with `scripts/running/run_generic_no_slurm.sh`.

Another important variable to change is `base`, which is defined
in every top-level script (in `scripts/running`). `base` determines where
files and folders should be written.

#### Names of corpora

The code assumes that `training_corpora` is set to `srf`, `focusnews` or both
(`training_corpora="srf focusnews"`).

The variable `testing_corpora` contains all
corpora for which evaluation results should be produced. `testing_corpora` can
contain four different names:

- `dev`: if `dev` is part of `testing_corpora`, it is assumed that you would like to 
translate and evaluate the held-out portion of the training data that is
used during training (for early stopping)
- `test`: translate and evaluate another held-out portion of the training data (different
from `dev`)
- `dev_unseen`: download (or link, see next section), preprocess, translate and evaluate the
development data we distribute separately (may not be available to you yet)
- `test_unseen`: download (or link, see next section), preprocess, translate and evaluate the
test data we distribute separately (may not be available to you yet)

`dev` and `test` are always split from the training data and preprocessed, regardless
of whether they are in `testing_corpora`. Their current size is 100 random samples
each.

`dev_unseen` and `test_unseen` are only downloaded (or linked) and preprocessed if they appear in `testing_corpora`.

### Downloading or linking to downloaded data

The scripts can either 1) download our training, dev and test data automatically and place them
in the folder `download`, or 2) place links to existing files (that you downloaded manually)
in `download`.

Regardless of whether to download or link, the variable `training_corpora` in the run script
determines which training corpora are considered. If `testing_corpora` contains `dev_unseen`
or `test_unseen`, it is assumed that you would like to download the dev or
test data we distribute separately.

#### Automatic download

If you'd like to download automatically from Zenodo, one or more environment variables
containing private tokens for accessing Zenodo must be set when calling a run script. For example,
to download FocusNews from Zenodo, a run script must be called as follows:

    ZENODO_TOKEN_FOCUSNEWS="your private token" ./scripts/running/dry_run_baseline_focusnews.sh

For SRF, both `ZENODO_TOKEN_SRF_POSES` and `ZENODO_TOKEN_SRF_VIDEOS_SUBTITLES` must be set, or exported beforehand.
These private tokens can be obtained by visiting the deposit websites on Zenodo and requesting
access. You will the be sent an email with a private link. The last portion of the private
link is an access token.

It is by design that these tokens cannot be defined as arguments in a run script since
they should not appear in commits or logs.

Also make sure that the variable `local_download_data` is not set in the run script.

#### Link to manual download

In the run script set the variable `local_download_data` to indicate where you downloaded the data.
The path must be the folder which contains the sub-folders `srf` and `focusnews`.
Each of these sub-folders in turn is expected to contain the sub-sub-folders
`videos`, `subtitles`, `openpose` and `mediapipe`.

Our separate dev and test data are always downloaded from a public URL, even if the variable `local_download_data` is set. If
download folders already exist, downloads are not repeated.

## Basic setup

Create a venv:

    ./scripts/setup/create_venv.sh

Then install required software:

    ./scripts/setup/install.sh

## Dry run

Try to create all files and run all scripts, but on CPU only and exit immediately without any actual computation:

    ./scripts/running/dry_run_baseline_focusnews.sh

By default, dry runs do not repeat the download step (meaning: dry runs prompt the user to delete all folders related to
a specific model, but by default do _not_ remove the download folder). To also repeat the download step, set
`repeat_download_step="true"`.

And a dry run for srf training data:

    ./scripts/running/dry_run_baseline_srf.sh

## Run a bilingual baseline

Train a baseline system for DSGS -> DE using focusnews data:

    scripts/running/run_baseline_focusnews.sh

A baseline using SRF data:

    scripts/running/run_baseline_srf.sh

## Baseline scores examples

From what we've seen so far, evaluation scores are extremely low, generally between 0.2 and 1.0 BLEU (varying simple top-level
settings achieves up to 1.0 BLEU). Here are the BLEU scores for some of the example running scripts:

| **training corpora** | **model name**     | **run script**                    | **test split from train** | **official dev** | **official test** |
|----------------------|--------------------|-----------------------------------|---------------------------|------------------|-------------------|
| focusnews            | baseline_focusnews | running/run_baseline_focusnews.sh | 0.231                     | 0.216            |                   |
| srf                  | baseline_srf       | running/run_baseline_srf.sh       | 0.354                     | 0.589            |                   |
| focusnews, srf       | baseline_both      | running/run_baseline_both.sh      | 0.074                     | 0.157            |                   |

"official dev" is the development data that was released separately, see https://www.wmt-slt.com/data. The baseline scripts
do not use the official dev data for training or validation.

The target labels of the official test data are not released yet.

## Making a submission

Submissions should use the [WMT XML format](https://github.com/wmt-conference/wmt-format-tools). We provide the test set sources in this
format. Here is how to download the source XML and combine it with baseline translations, using the system `baseline_focusnews` as 
an example:

    wget https://files.ifi.uzh.ch/cl/archiv/2022/easier/wmtslt/test/dsgs-de/slttest2022.dsgs-de.sources.xml

    wmt-wrap \
        -s slttest2022.dsgs-de.sources.xml \
        -t translations/dsgs-de/baseline_focusnews/test.trg \
        -n baseline_focusnews -l de \
        > slttest2022.dsgs-de.hypo.de.xml

Then upload the resulting file `slttest2022.dsgs-de.hypo.de.xml` on the OCELoT submission platform.

## Ideas for improving this baseline

- find bugs :-)
- use both SRF and focusnews training data in the same system
- also use monolingual subtitles (around 2000 files)
- openpose do not take the first person's keypoints, be more elaborate
- different / better normalization or scaling of poses
- augmentation, frame dropout from pose_format
- deal with framerate differences better or make use of them for augmentation
- graph encoder for poses
- various forms of pre-training
