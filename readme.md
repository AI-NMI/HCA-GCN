# HCA-GCN

Reference implementation of the hierarchical multi-view brain-connectivity
fusion model used for rs-fMRI neuropsychiatric-disorder classification.

This repository contains model, training, evaluation, and split-generation code.
It does not contain raw fMRI, derived ROI signals, connectivity matrices, or
participant metadata.

## Reproducibility protocol

The implementation follows the revised manuscript protocol.

- Cross-validation is performed at the subject level.
- A subject appears in exactly one held-out test fold.
- Every fold has disjoint training, validation, and test subject IDs.
- All normalization statistics are fitted on the training partition only.
- The checkpoint is selected by the minimum validation loss.
- The held-out test fold is evaluated only after checkpoint selection.
- ACC, AUC, SEN, and SPE are computed from that single checkpoint.
- Binary predictions use one fixed probability threshold of 0.5.
- AUC is computed from the unthresholded class-1 probabilities.
- Per-subject out-of-fold probabilities and per-fold confusion matrices are
  saved for audit and independent metric recomputation.

## Input representations

For every subject, five aligned inputs are expected:

1. ROI time series;
2. nonlinear cLSTM-based Granger connectivity;
3. linear pairwise Granger connectivity stored as raw p-values;
4. Pearson functional connectivity;
5. Spearman functional connectivity.

Connectivity matrices and time-series files must share the same deidentified
subject ID. The loader intersects IDs across all modalities and fails on
duplicates, missing keys, unexpected shapes, or overlap between partitions.

The imaging preprocessing and connectivity estimation are upstream steps. In
the experiments, the first 10 fMRI volumes were removed and a continuous
segment was retained: 100 time points for ABIDE II and ADHD-200, and 140 time
points for COBRE. The cLSTM estimator used hidden dimension 64 and context
length 10. Linear Granger estimation used lag 2 and saved pairwise p-values.

## Model variants

Set model.variant in the YAML configuration:

- full: temporal branch and two learned graphs;
- m4_no_temporal: removes the temporal branch;
- m5_single_fused_graph: combines learned FC and EC into one graph after
  row-wise z-score normalization;
- m6_original_connectivity: disables adaptive graph construction and uses the
  original association- and causality-informed matrices after row-wise z-score
  normalization.

The graph node representation is 32-dimensional. The temporal Transformer
hidden dimension is 64.

## Data layout

Set the local data root without editing source code:

~~~bash
export HCA_GCN_DATA_ROOT=/path/to/private/derived_data
~~~

The example configuration expects:

~~~text
HCA_GCN_DATA_ROOT/
+-- timeseries/
|   +-- control/
|   +-- patient/
+-- connectivity/
    +-- nonlinear_gc/{control,patient}/
    +-- linear_granger_p/{control,patient}/
    +-- pearson/{control,patient}/
    +-- spearman/{control,patient}/
~~~

Each directory contains one MATLAB file per subject. Default variable names
are ROISignals, GC_est, Granger, Pearson, and Spearman; they can be changed in
the YAML file.

## Installation

Python 3.8 or newer is recommended.

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

## Run

From the repository root:

~~~bash
python main.py --config setting/abide_HCAGCN.yaml
~~~

For ABIDE II or ADHD-200, use time_points: 100 in the data section and
time_series: 100 in the model section. COBRE uses 140 for both values.

The COBRE cohort contains 145 subjects. The revised configuration generates
held-out fold sizes of 29, 28, 29, 29, and 30. To publish an exact split, set
data.split_file to a committed JSON file; otherwise the deterministic generator
writes the generated IDs under outputs/splits/.

## Outputs

Every run writes best_by_validation_loss.pt, history.json, metrics.json,
test_predictions.csv, held_out_graphs.npz, cross_validation_summary.json, and
oof_predictions.csv. metrics.json includes TP, FN, TN, FP, ACC, AUC, SEN, and
SPE.

The outputs directory is ignored by Git. Review all subject identifiers before
publishing split files and predictions.

## Repository hygiene

Private data paths, credentials, raw data, generated checkpoints, logs, caches,
and local editor files must not be committed. A license has intentionally not
been selected automatically; the authors should choose and add the intended
open-source license before public release.
