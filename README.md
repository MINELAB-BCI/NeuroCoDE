# NeuroCoDE

Few-shot 3-class motor imagery classification (reaching / multigrasp / twist) with:

- 25 channels mapped to a 5×5 grid
- mu (8–13 Hz) and high-gamma (50–70 Hz) band embeddings
- **learnable band weights** fused **before** prototype construction (early fusion)
- cosine similarity to class prototypes

Each participant was recorded in three sessions. Released checkpoints are provided for two conditions that differ **only in the prior sessions used for supervised pretraining**. Both conditions are fine-tuned and evaluated on the same target session (session 1), with identical query trials.

| condition | model in the paper | supervised pretraining (prior sessions) | fine-tuning + test (target session) |
|---|---|---|---|
| `two_prior_sessions` | B2 | session 2 + session 3 | session 1 |
| `one_prior_session` | A1 | session 2 | session 1 |

For each subject, the fine-tuning and held-out query trials are class-balanced and **disjoint**:

- 80 trials per class for fine-tuning
- 20 trials per class for held-out evaluation

Reported numbers are the mean ± standard deviation over **100** random support draws for the 1-, 5-, and 25-shot settings.

## Data availability

MI EEG files are **not included in this repository**. Please contact the authors to request access to the data.

After obtaining the data, set `DATA_ROOT` to the folder containing the BrainVision files of all sessions:

```bash
export DATA_ROOT=/path/to/MI_EEG_data
```

```text
<DATA_ROOT>/
├── session1_sub1_reaching_MI.vhdr / .vmrk / .eeg
├── session1_sub1_multigrasp_MI.vhdr / .vmrk / .eeg
├── session1_sub1_twist_MI.vhdr / .vmrk / .eeg
├── session2_sub1_...
├── session3_sub1_...
└── ...
```

Inference needs only the session-1 (target) files. Training from scratch also needs the prior sessions (session 2, and session 3 for `two_prior_sessions`).

## Setup

From the repository root, install the required packages:

```bash
pip install -r requirements.txt
```

## Reproduce released numbers (inference only)

Only the target-session files are required for inference. The prior sessions are not loaded.

Set the data root and run inference from the repository root:

```bash
export DATA_ROOT=/path/to/MI_EEG_data

cd code
python run_infer.py --condition one_prior_session --data_root "$DATA_ROOT"
python run_infer.py --condition two_prior_sessions --data_root "$DATA_ROOT"
```

Optional: evaluate only selected subjects using `--subjects`.

```bash
export DATA_ROOT=/path/to/MI_EEG_data

cd code
python run_infer.py \
  --condition one_prior_session \
  --data_root "$DATA_ROOT" \
  --subjects sub1,sub2
```

Each subject is compared against:

```text
weights/<condition>/subX/sweep_result.json
```

A `[MATCH]` line means that the reproduced 1-, 5-, and 25-shot mean accuracies agree with the released results within 0.05 percentage points.

## Train from scratch

| condition | prior sessions loaded for supervised pretraining | fine-tuning + test |
|---|---|---|
| `two_prior_sessions` | `session2_subX_*`, `session3_subX_*` | `session1_subX_*` |
| `one_prior_session` | `session2_subX_*` | `session1_subX_*` |

Set the data root and run training from the repository root:

```bash
export DATA_ROOT=/path/to/MI_EEG_data

cd code
python run_train.py --condition one_prior_session --data_root "$DATA_ROOT" --retrain
python run_train.py --condition two_prior_sessions --data_root "$DATA_ROOT" --retrain
```

Checkpoints are written to:

```text
weights/<condition>/subX/band_prototype_model.pth
```

Excel summaries are written to:

```text
results/two_prior_sessions.xlsx
results/one_prior_session.xlsx
```

## Repository layout

```text
band-prototype-eeg/
├── README.md
├── requirements.txt
├── .gitignore
├── code/
│   ├── global_config.py
│   ├── band_prototype_main.py
│   ├── run_train.py
│   └── run_infer.py
├── weights/
│   ├── two_prior_sessions/
│   │   ├── sub1/
│   │   │   ├── band_prototype_model.pth
│   │   │   └── sweep_result.json
│   │   ├── sub2/
│   │   └── ...
│   └── one_prior_session/
│       ├── sub1/
│       │   ├── band_prototype_model.pth
│       │   └── sweep_result.json
│       ├── sub2/
│       └── ...
└── results/
    ├── two_prior_sessions.xlsx
    └── one_prior_session.xlsx
```

### Source files

- `code/global_config.py`: global configuration, including random seed 42
- `code/band_prototype_main.py`: model, dataset, training, and evaluation implementations
- `code/run_train.py`: supervised pretraining and few-shot fine-tuning pipeline
- `code/run_infer.py`: inference and released-result reproduction

## Tools and source code references

### sLORETA

[sLORETA](https://www.uzh.ch/keyinst/loreta) (Standardized Low-Resolution Brain Electromagnetic Tomography) was used for EEG source localization and signal-quality validation.

sLORETA is a neuroimaging technique that estimates the three-dimensional distribution of electrical activity in the brain from scalp-recorded EEG signals. It addresses the EEG inverse problem by inferring the underlying neural sources and standardizes the estimates to reduce localization bias. Although it provides relatively low spatial resolution, sLORETA produces stable source-activity maps and is widely used in clinical and neuroscience research.

In this research, sLORETA was employed to assess the quality of the recorded EEG signals and verify that motor imagery activity occurred in motor-related cortical regions. The validation results support the feasibility of EEG-based robotic arm control using the selected motor-cortex channels.

### BBCI Toolbox

The [BBCI Toolbox](https://github.com/bbci/bbci_public) was used as a reference for real-time EEG processing and brain–computer interface applications.

## Notes

- Released-result inference uses only the saved `.pth` checkpoints.
- Do not pass `--retrain` when reproducing the released results.
- Supervised pretraining data are not loaded during inference.
- On the first run, EEG signals are filtered and written to the local `cache/` directory.
- The `cache/` directory is excluded from version control.
- Subsequent runs reuse the cached data.
- Small floating-point differences may occur between GPU and CPU execution.
- `run_infer.py` allows a tolerance of ±0.05 percentage points.
