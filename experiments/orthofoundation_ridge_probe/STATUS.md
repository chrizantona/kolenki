# EXP-OF-003: slot-centered Ridge — completed

**Actual CPU fit completed and independently validated.** Weak870 AUC **0.7758808435**;
Gold58 diagnostic AUC **0.7719155174**. One CV fit on3479 weak studies and one
production fit on4349; Gold gradients/transform/model selection0. Parent9.021s,
peak1.265GB, no encoder/GPU/API/submission. [Actual report](actual_probe_report.json),
[parent proof](actual_parent_report.json), [independent actual audit](independent_actual_Ridge_validation.json),
[preparation review](independent_preparation_review.json).

Historical preparation follows; its requirements were fulfilled before actual acceptance.


The probe reuses the independently sealed complete OrthoFoundation bank. It averages the four vectors separately within each observed slot, fits each slot's mean/std on that model's gradient studies only, standardizes with a0.001 floor, and appends six mask indicators. Ridge uses alpha1000, an intercept and the Cholesky solver on weak-label logits clipped at1e-4. Prediction applies sigmoid once. There is one CV fit on3479 weak studies and one production fit on4349;870 holdout studies are excluded from the CV scaler/model. Production uses all4349 weak studies, including the former CV holdout;58 Gold are excluded from both scalers/models.

This tests whether fixed, slot-specific linear readout can recover information the attention head missed. It introduces no encoder, MRI decode, GPU, hyperparameter search, Gold-driven model choice or submission. Closed-form fitting uses no stochastic minibatch order; seed42 is recorded for the experiment. All original source8, attention config and inference sources remain unchanged.

This is a representation readout diagnostic: pooling, slot standardization, model capacity, optimizer and target-logit squared-error loss change together from attention/BCE. It is not a causal head-only ablation. A weak result cannot prove that the original features contain no useful information; K4 mean pooling may discard slice-specific signals and positions.

CPU execution must run in the existing owned process-group guard with a300-second deadline, two BLAS threads and a private output directory. Python checks elapsed time and peak resident memory below2GiB during validation, before/after each fit and export. A completed probe report requires a successful external parent receipt too; forced termination cannot guarantee Python rollback. Scalers, coefficients and raw predictions remain private.

The CLI requires explicit external ROOT bank/seal SHA values and refuses changed bank bytes before fitting. It reads each CSV/NPZ as one immutable byte snapshot, verifies the SHA, and parses that same snapshot. Expected complete source-bank SHA is supplied by ROOT, never derived from the probe's current bank.

After independent review, ROOT can execute [probe_orthofoundation_ridge.py](../../scripts/probe_orthofoundation_ridge.py) with `--feature-bank`, `--source-bank-sha256`, `--root-seal`, `--root-seal-sha256`, `--global-metadata`, `--output-dir` and `--execute-reviewed-cpu-probe`. The pinned inputs are the completed HTTP V2 attention `run/features`, ROOT's full merge seal and the verified global metadata directory. The source bank is `acc8b0c85639833cb51fd101de6b0fe605667150e88d2fdb7476b94ef6f46f74`; the ROOT seal is `7bd85a75523a95e071c91b831f631d1a9674612e7443693df60a8cf19828a6ee`.
