# Multimodal Keystroke + Glove-Kinematics Authentication

Code accompanying the manuscript **"Multimodal User Authentication via Keystroke-Triggered
Fusion of Finger Pressure and Hand Kinematics"** (IEEE Access, manuscript ID Access-2026-29299).

The raw recordings are not published, but every run's per-fold output and the per-keystroke
scores behind the headline result are, so the tables and the statistical tests in the paper
can be checked from `results/` alone — no GPU, no raw data.

## The method in one paragraph

Each key press triggers extraction of a 600 ms window of glove signal. The ten pressure
channels and the nine inertial channels are normalized per window, the pressure level that
normalization removes is restored as ten extra channels, and the two modalities are then
encoded by **separate CNN-LSTM branches** whose embeddings are concatenated before a single
classification head. Scores are averaged over $W$ consecutive keystrokes; $W = 3$ is recommended.

The scale of the appended block is the part that matters. Those ten channels are mu/sigma and
run far larger than the Z-scored nineteen, so whatever convolution mixes them is dominated by
them: a single encoder over all 29 reaches 4.20% at W = 3 against 0.53% for two branches.
Standardizing the block closes the gap. A single encoder then reaches 0.69%, which nine folds
cannot separate from two branches (p = 0.84). Separate encoding achieves that structurally, by
keeping the inertial encoder out of the competition, and explicit standardization achieves it
directly. Either works, and neither is better.

## Headline results

Leave-One-Impostor-Out, cross-domain (trained on desktop, tested on laptop), the unseen
impostor rotated over all nine remaining participants, three seeds per fold.

| Smoothing window | EER | FRR at FAR = 0.1% | Decision latency |
|---|---|---|---|
| W = 3 (recommended) | **0.53 ± 0.42%** | 3.09% | 0.77 s |
| W = 5 | 0.18 ± 0.22% | 0.68% | 1.23 s |

W = 3 is recommended because it is the largest smoothing window at which every fold still
produces a measurable error rate. At W = 5 one fold of nine observes no error at all, and the
test-set resolution is 1/3950 ≈ 0.025%, so larger windows increasingly measure the size of the
test set rather than the system.

## Layout

```
Proposed-TwoBranch.ipynb          the proposed model; the headline evaluation
Architecture-Rotated-29ch.ipynb   the architecture comparison, nine folds, 189 runs
Scale-vs-Architecture.ipynb       is the margin the encoder or the scale of the level block
Sensor-Confound-Control.ipynb     magnetometer and pressure level removed, one at a time
Fusion-Design-Test.ipynb          four ways of fusing two modalities
Modality-Ablation.ipynb           per-modality contribution, single-stream encoder
Normalization-Test.ipynb          why the pressure channels underperform
Mixed-Normalization-Test.ipynb    restoring it as a ratio; also negative
Window-Sweep-Rotated.ipynb        feature-window sensitivity, ten lengths, 270 runs

retuned/                          the partially nested search and its outcome (see below)
  Optuna-Nested-TwoBranch.ipynb     the search: 200 trials, reported session withheld
  Proposed-TwoBranch-Retuned.ipynb  the headline evaluation re-run under what it chose
  Window-Sweep-Rotated-Retuned.ipynb  the window sweep, likewise re-run
  results/                          its outputs, kept apart from the published ones

results/                          one row per run, plus per-keystroke scores
```

## Which experiment produced which number

| Directory | Notebook | Backs |
|---|---|---|
| `two-branch/` | `Proposed-TwoBranch.ipynb` | the headline LOIO table, the smoothing/FRR/DET figures, the capacity control, the proposed row of the architecture table |
| `architecture-rotated-29ch/` | `Architecture-Rotated-29ch.ipynb` | every row of the architecture table, and the latency figure |
| `scale-vs-architecture/` | `Scale-vs-Architecture.ipynb` | the two-branch cells of the level-and-encoder table |
| `confound-control/` | `Sensor-Confound-Control.ipynb` | the channel-control table, the level-free cells of the level-and-encoder table, and the 0.58 impostor-SD figure in the limitations |
| `fusion-design/` | `Fusion-Design-Test.ipynb` | the pressure-only, IMU-only and two-branch rows of the modality ablation, and the fusion-design figure |
| `modality-ablation/` | `Modality-Ablation.ipynb` | the 19-channel single-stream ablation, and the r = −0.085 / 0.976 capture correlations |
| `mixed-normalization/` | `Mixed-Normalization-Test.ipynb` | the 29-channel single-stream row, and the r = 0.977 / −0.184 capture correlations |
| `normalization-test/` | `Normalization-Test.ipynb` | the 2.69% a pressure-only model reaches at W = 5 once the absolute level is retained |
| `pressure-level/` | — | the between-recording shift of each channel, and the drift of the pressure level against the spread between people |
| `window-sweep-rotated/` | `Window-Sweep-Rotated.ipynb` | the feature-window sensitivity table and figure |
| `retuned/results/` | `retuned/*.ipynb` | the partially nested search of Section III-E: the 101,634-parameter configuration it selected, and the null result that keeps the published one |

Each directory holds a `*_raw*.csv` with one row per (arm, fold, seed) carrying **every**
smoothing window from W = 1 to W = 10, not only the three the paper prints, and a
`*_summary.csv` aggregated the way the paper reports it. `two-branch/loso_scores.npz` holds
the per-keystroke score of every run of the proposed model with its label and recording-unit
group, so any smoothing window, any decision threshold, the DET curves and the FRR at any
operating point can be recomputed from it.

Several arms reach zero observed errors at larger W. The `at_floor` column of
`two-branch/loso_summary.csv` counts them: a zero there means no error was observed at the
resolution of the test set, not that the error rate is zero.

## Running the experiments

The notebooks expect the raw per-participant CSV files under `data/<participant>/` and a
`participants_local.json` mapping recording folders to the anonymized `P01`–`P10` labels.
Neither is published (see *Data availability*). With those in place, run
`Proposed-TwoBranch.ipynb` first: it produces the headline result the other notebooks check
themselves against.

Several notebooks carry **runtime assertions rather than assumptions**.
`Window-Sweep-Rotated.ipynb` runs its 600 ms arm first and refuses to continue unless it
reproduces the headline table. `Fusion-Design-Test.ipynb` and the normalization notebooks
re-run their single-modality arms rather than quoting them, so a known value has to be
reproduced on every run. If one of these checks fails, the run is wrong, not the check.

## The partially nested search, and what it settled

`retuned/Optuna-Nested-TwoBranch.ipynb` runs 200 trials against the published two-branch
model with the genuine user's laptop session — the cross-domain test set behind every number
in the paper — withheld from both the objective and the search's training data. It searches
the window length, the sequence length, the hidden size and the learning rate, and also the
dropout rate, the LSTM depth and the weight decay. It objects to itself in two ways, both
recorded in `retuned/results/optuna-nested/best_params.json`: the inner impostors also appear as outer
folds of Table 1, so it is *partially* nested rather than nested; and trials were ranked at 12
epochs on half the windows with one seed, and no trial was refitted at the full budget.

It has now been run. 115 trials completed, 85 were pruned, and the best trial selected a
markedly smaller model:

| Parameter | Published | Selected by the search |
|---|---|---|
| `target_len` (L) | 45 | 35 |
| `hidden_dim` (d) | 64 | 32 |
| `num_layers` | 2 | 1 |
| `dropout` (p) | 0.395 | 0.129 |
| `lr` | 1.814e-3 | 3.895e-4 |
| `weight_decay` | 0.01 | 1.186e-5 |
| parameters | 230,146 | 101,634 |

`retuned/Proposed-TwoBranch-Retuned.ipynb` then re-runs the full rotated evaluation under that
configuration — same folds, same three seeds, same protocol — and the outcome is a null
result:

| W | Published config | Retuned config | Better in | Wilcoxon p |
|---|---|---|---|---|
| 1 | 1.55% | 1.43% | 6/9 | 0.20 |
| 3 | 0.53% | 0.56% | 5/9 | 1.00 |
| 5 | 0.18% | 0.18% | 4/9 | 0.67 |

Halving the depth and the width and cutting the parameter count by 56% changes nothing that
nine folds can detect. The paper keeps the original configuration, because it is the one every
other experiment in `results/` was run under, and reports this search as the check that the
configuration is not an artefact of how it was chosen. Nothing in the manuscript's tables depends on
`retuned/`.

## Data availability

**Raw participant data is intentionally NOT included.** The study collected keystroke timing
and glove sensor recordings from ten human participants, and this raw, personally identifiable
data is not being made public, consistent with the informed-consent protocol under which it
was collected. It is available from the corresponding author on reasonable request, subject to
participant consent. Participant identifiers in the notebooks are anonymized to `P01`–`P10`.

`participants_local.json`, which maps those labels to the real recording folders, is listed in
`.gitignore` and must never be committed: the identifiers are directly identifying. The
notebooks fall back to the anonymized roster whenever the file is absent, which is the state
this repository should always be in.

The data-collection hardware and firmware are not part of this release.

## Requirements

PyTorch, Optuna, and the standard scientific-Python stack (numpy, pandas, scipy,
scikit-learn, matplotlib):

```
pip install -r requirements.txt
```

Reproducing the tables from `results/` needs none of PyTorch, Optuna or a GPU
— only numpy, pandas and scipy.

## Citation

If you use this code, please cite the associated manuscript (citation details to be added once
the final publication information is available).
