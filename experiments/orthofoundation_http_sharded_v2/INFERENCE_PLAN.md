# Future Ortho inference seam

This is a source-reviewed plan, with no inference implementation or new
submission. The complete feature bank and its fixed-final, 12-epoch production
attention `head.pt` do not exist yet. Partial-shard smoke heads are extraction
checks, not production models.

The submitted 0.944 candidate combines the public stack with our three-reader
ConvNeXt average using 70:30 percentile ranks, then ranks that combination again.
A future, separate Ortho inference child should write `_ortho.csv` beside that
final combination stage. Keep the
existing `_own.csv` and candidate intact. Any blend or submission requires a
separate decision; this plan makes no score-gain claim.

## Raw MRI to the same feature recipe

1. Reuse [preprocess.py](../../src/convnext_reader/preprocess.py)::`load_series`
   for each eligible test series, passing its metadata `Anatomical_Plane` as
   `plane_name`. It applies DICOM rescaling/MONOCHROME1 handling, canonical LPS
   orientation and slice order, a centered 153.6 mm field at 0.4 mm/pixel, and
   zero-padding to 384×384. It keeps at most 64 uniformly selected slices before
   the per-series 1st/99.5th-percentile window, then rounds to uint8. Memoize these
   volumes in owned temporary storage. Use decoded, capped slice counts, not
   directory file counts; volumes with fewer than three slices are unusable.
2. Reuse [data.py](../../src/orthofoundation/data.py)::`build_table`,
   `slice_indices` and `SliceStudyDataset`. Select the deepest usable series per
   plane/fat-suppression slot, breaking equal-depth ties by series UID. Preserve
   six slots, four centers from 4%–96% of the available depth, normalized
   positions `center/(n-1)`, and the boolean slot mask. Missing slots have zero
   features/positions; a study with no usable slot must fail.
3. Use `data.make_views` exactly: one grayscale slice repeated into RGB,
   centered **353×353** crop from the 384 cache (`round(384*0.92)`), antialiased
   bilinear resize to 224, and ImageNet normalization. This is our frozen
   preprocessing choice, not an exact reproduction of the authors' MRI input.
4. Reuse [encoder.py](../../src/orthofoundation/encoder.py)::`load_encoder`
   and `encode` with the strict released checkpoint, frozen/eval DINOv3-L/16,
   `author_no_rope`, FP16 CUDA autocast and image batches of 16. Preserve the
   per-study extraction grouping. Match training storage by rounding the
   1024-dimensional CLS features to **float16**, then converting to float32 for
   the head; direct float32 CLS input would change the feature recipe.
5. Load only the future production `head.pt` with strict finite state, exact
   labels/config, external file SHA and training receipt: 12 fixed epochs,
   4,349 weak gradient studies, all 58 Gold studies excluded, no Gold checkpoint
   selection. Reuse [heads.py](../../src/orthofoundation/heads.py)::`StudyHead`
   in eval/FP32 and [train.py](../../src/orthofoundation/train.py)::`predict`
   semantics, applying sigmoid once. Refuse `head_weak_holdout.pt` and smoke
   checkpoints.

## Existing inference is not interchangeable

[ConvNeXt infer.py](../../src/convnext_reader/infer.py) decodes raw DICOMs
without persisting a canonical test-volume cache. Its series ranking uses raw
file counts and input-order ties, omits the metadata-plane fallback, and uses
K=12 windows with **three neighboring slices as RGB**. Ortho uses K=4 single
grayscale slices replicated as RGB. ConvNeXt's affine-grid crop at 0.92 also
differs numerically from Ortho's literal 353-pixel crop/antialiased resize.
Other public-stack rendered caches use different crop/percentile conventions;
they cannot be assumed equivalent to these canonical volumes.

## Pins and future acceptance

| Frozen item | SHA256 |
| --- | --- |
| Canonical `preprocess.py` | `68601ad84ec3a0f030b5c2d37cc0d657cd3e12b17f7130e7385fba9648d324b6` |
| Ortho `data.py` | `f3b63ab922f0452c065126dc2ef1c87442a0b988dff29d25c201666a8a5310aa` |
| Ortho `encoder.py` | `86defa5f943bfc161603c7b377bbf08a9d5c2a270d01b1df5c039ab59cff7ecb` |
| Released encoder checkpoint, 1,213,056,638 bytes | `385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9` |
| [V2 runner](../../scripts/orthofoundation_http_v2/sharded_frozen_extract.py) | `86265241778c40a3822c2a8d11f4f3f746a48b47101def338120dd86937ee4f2` |
| [V2 attention config](../../configs/orthofoundation_frozen_http_v2.yaml) | `f3c4334f97df527d22dd2b0da3b6220ec466c1333efe7e99a471b955a74ba919` |

All eight scientific source pins are recorded in the
[preparation receipt](preparation_receipt.json). The parent training feature
identity is `7f0e16dc1c34e8c0920f567db2ebb68f55c470c09ce77dd7c0aa238cd5ec6c77`.
Test metadata, selection and source hashes require a separate test provenance
identity referencing that parent. The training identity's 4,407-study global
cache plan must not be claimed for demonstration or hidden test studies.

Before integration, verify canonical decode parity against known cached
series, then test masks/K4 positions/features/head replay within a declared
numerical tolerance. Check exact submission study coverage, ordered 12 labels,
finite probabilities in [0,1], and record runtime/memory. No such model replay
or inference run has been performed for this plan. Prediction rows and clinical
identifiers remain private.
