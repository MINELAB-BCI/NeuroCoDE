# Band-prototype EEG (early fusion)

Few-shot 3-class motor imagery classification (reaching / multigrasp / twist) with:

- 25 channels mapped to a 5×5 grid
- mu (8–13 Hz) and high-gamma (50–70 Hz) band embeddings
- **learnable band weights** fused **before** the prototype (early fusion)
- cosine similarity to class prototypes
- cue-locked window **0–4 s**, random seed **42**

Released checkpoints were finetuned and tested on **w/ robotic arm MI data** and **w/o robotic arm MI data**. The two folders differ only in which sessions were used for **supervised pretraining**.

| condition | pretrain | finetune + test |
|---|---|---|
| `with_robotic_arm` | Motor imagery data | w/ robotic arm MI data |
| `without_robotic_arm` | Motor imagery data | w/o robotic arm MI data |

Train and test trials on session 1 are class-balanced and **disjoint** (80/class train, 20/class held-out query). Reported numbers are the mean ± std over **100** random support draws (1 / 5 / 25-shot).

MI EEG files are **not** in this repository. 요청하세요.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce released numbers (inference only)

Session 1 files are enough. Pretraining sessions are not loaded.

```bash
cd code
python run_infer.py --condition without_robotic_arm --data_root /path/to/GIGA/RawData
python run_infer.py --condition with_robotic_arm --data_root /path/to/GIGA/RawData
```

Optional: `--subjects sub1,sub2`

Each subject is compared against `weights/<condition>/subX/sweep_result.json`. A `[MATCH]` line means the 1/5/25-shot means agree within 0.05 percentage points.

## Train from scratch (same pipeline as the released weights)

`without_robotic_arm` needs session 1 + w_pretrain_data.  
`with_robotic_arm` needs session 1 + w_pretrain_data.

```bash
cd code
python run_train.py --condition without_robotic_arm --data_root /path/to/GIGA/RawData --retrain
python run_train.py --condition with_robotic_arm --data_root /path/to/GIGA/RawData --retrain
```

Checkpoints are written to `weights/<condition>/subX/band_prototype_model.pth`.  
Excel summaries go to `results/<condition>.xlsx`.

## Layout

```
band-prototype-eeg/
  README.md
  requirements.txt
  .gitignore
  code/
    global_config.py          # seed = 42
    band_prototype_main.py    # model, dataset, train, eval
    run_train.py
    run_infer.py
  weights/
    with_robotic_arm/sub1..sub25/
      band_prototype_model.pth
      sweep_result.json
    without_robotic_arm/sub1..sub25/
      ...
  results/
    with_robotic_arm.xlsx
    without_robotic_arm.xlsx
```

## Tools and Source Code References
- sLORETA(Standardized Low-Resolution Brain Electromagnetic Tomography): Source localization for EEG signal analysis. Learn more at [sLORETA](https://www.uzh.ch/keyinst/loreta).
  - sLORETA is a neuroimaging technique that estimates the three-dimensional distribution of electrical activity in the brain based on EEG measurements. It addresses the inverse problem by inferring the sources of the scalp-recorded electrical signals and standardizes the results to reduce localization bias. Although it provides low spatial resolution, sLORETA delivers a stable and reliable map of brain activity, making it widely used in both clinical research and neuroscience.
  - In this research, sLORETA was employed to assess the quality of the recorded brain signals and to verify that motor imagery was indeed occurring in the motor cortex. The validation results clearly demonstrate that the observed EEG patterns accurately reflect the neural activity associated with the cognitive processes underlying motor imagery, effectively supporting the feasibility of EEG-based robotic arm control using only the channels corresponding to the motor cortex.

- BBCI Toolbox: For real-time EEG processing and BCI applications. Available at [BBCI Toolbox](https://github.com/bbci/bbci_public).

## Notes

- Inference uses the saved `.pth` only. Do not pass `--retrain` if you want the uploaded numbers.
- First run filters EEG and writes `cache/` (gitignored). Later runs reuse that cache.
- GPU vs CPU can produce tiny floating-point differences; `run_infer.py` allows ±0.05%p.
