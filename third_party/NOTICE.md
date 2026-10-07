# Third-party code and model attribution

The repository root MIT license applies to our original additions. It does not replace upstream licenses or the competition data terms.

## GoodPJ ConvNeXt reader

`src/convnext_reader/knee.py`, `preprocess.py`, `infer.py`, `make_labels_reference.py` are copied from the public GoodPJ reader dataset, version 1. The dataset declares **Apache 2.0**. `train.py` is our modified Kaggle fp16/resumable training port. `zip_cache_loader.py` is our cache integration code. Source provenance is recorded in `source_manifest.json` and the experiment evidence.

- Author: goodpjw2008 / GoodPJ
- Source: https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader
- Public stack notebook: https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944?scriptVersionId=355300588
- License text: [Apache 2.0](LICENSE-APACHE-2.0.txt)

The entire upstream community ensemble is referenced by pinned notebook URLs; this repository does not claim authorship of those models or include their weights.

## OrthoFoundation and DINOv3

- OrthoFoundation code/checkpoint: https://github.com/ytrsk/OrthoFoundation
- DINOv3: https://github.com/facebookresearch/dinov3
- DINOv3 license: https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md

The OrthoFoundation project code is MIT, but its DINOv3-derived model and the official DINOv3 implementation carry upstream DINOv3 terms. Model weights and upstream source archives are held in a private Kaggle asset dataset with the corresponding notices; they are not relicensed by this repository or stored in Git.

Raw RSNA competition data, reports, labels, model checkpoints and credentials are excluded from Git.
