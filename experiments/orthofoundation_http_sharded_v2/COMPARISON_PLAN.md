# Gold comparison plan — no Ortho head result yet

The saved fixed-final ConvNeXt EMA reader after two epochs has 58 unique Gold
rows and 12 finite logit columns. Its study set and expert labels exactly match
the Gold58 population specified for Ortho. The sorted Gold-ID hash is
`3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea`.
No medical rows, study IDs or prediction CSV are published here.

The baseline prediction file SHA256 is
`c9ef127918f1e7758853327ed1d0d2edb4ae3be38eb05935f60e4390585a7cf5`;
its aggregate metric receipt SHA256 is
`9e536a2e3e3bb72c7df31772ba99b218b3bded85f15064c52baaae60ea6a3cae`.
Saved Gold macro AUC is **0.9111602412161779**. The checkpoint is the fixed-final
EMA with SHA256
`2557782563a0b416cf85c7bf98a0caaf685dc4f991409b184b8b118730dfa25a`.
[Training report](../convnext_fullweak_2ep/training_report.json) and
[provenance](../convnext_fullweak_2ep/provenance.json) document the completed run.

This reference is one reader from the submitted ensemble. The full ensemble's
public score **0.944** is a separate observation. Its locally saved inference
CSVs contain three demonstration test studies and zero Gold overlap; they
cannot supply a Gold comparison of the complete ensemble. The reader was
warm-started from a public task-trained checkpoint whose original supervised
exposure is not certified clean OOF. Its additional training used all 4,349 weak
studies, so an Ortho weak-fold0 comparison against this reader would not be fair
held-out validation. No saved ConvNeXt weak OOF predictions are available.

After completed Ortho production heads and their receipts are verified:

- Require the exact shared 58-study set, label order, official expert labels,
  finite probabilities and matching checkpoint/source provenance. Seal the
  complete source-bank SHA before the separate mean/max comparison and retain
  its checkpoint replay checks.
- Report all 12 per-label AUCs, macro AUC, positive/negative support and paired
  study-bootstrap uncertainty for differences. Gold is small; the rarest label
  has only nine positives. Report bootstrap samples with undefined AUC rather
  than concealing them.
- Examine per-label rank correlation and overlap of errors. Keep case-level
  discrepancies private; public reports contain aggregate counts only.
- ConvNeXt scores are raw logits. AUC can use them directly; apply sigmoid
  exactly once before probability errors or blending with Ortho probabilities.
  If used, predeclare one fixed 50:50 probability blend as an ensemble diagnostic.
  Do not fit weights, choose checkpoints or select a blend recipe on Gold.

Gold comparisons remain diagnostic. They cannot establish clean generalization,
an Ortho leaderboard score or an expected score gain. Ortho head predictions,
full-bank completion and any comparison metrics still require actual execution.
