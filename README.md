# AADMN — Adaptive Attention-based Dynamic Mesh Networks

**A graph-based detection re-scoring approach for crowded scenes.**

> **Status:** Pre-publication. A manuscript based on this work is in preparation.
> The source code is **not** included in this repository — see
> [Code availability](#code-availability). All rights reserved
> (see [LICENSE](LICENSE)).

---

## Overview

AADMN investigates whether **graph-based scene context** can improve object detection
in crowded scenes. A lightweight **YOLOv8n** detector provides candidate detections; a
**Graph Neural Network** re-scoring head then refines each detection's confidence using
a scene graph built from the detector's own outputs.

The study runs a controlled ablation between two re-scoring architectures under matched
training conditions:

- **SceneGNN** — a plain GraphSAGE model (uniform neighbour aggregation).
- **SceneGAT** — a Graph Attention Network (learned neighbour weighting).

The central finding is a **substantive negative result**: adaptive attention does *not*
improve over uniform aggregation on this task, and mechanistic analysis of the trained
attention weights explains why.

![AADMN pipeline](figures/architecture.png)

---

## Key findings

- **Strong baseline detector.** YOLOv8n trained on CrowdHuman reaches
  **mAP50 = 0.819** and **mAP50–95 = 0.515** on the validation set (4,370 images).
- **Graph context helps.** A scene-graph re-scoring layer (GraphSAGE, 11-feature nodes,
  k-NN edges with k = 8) improves true/false-detection separation accuracy by
  **+12.54 percentage points** over raw detector confidence (0.7340 → 0.8594).
- **Attention does not earn its place.** Across two feature-set sizes (5 and 11
  features), the GAT consistently scored *slightly below* the plain GNN. Richer features
  helped the GNN but not the GAT.
- **The negative result is mechanistic, not an artefact.** The trained GAT's first-layer
  attention weights are highly variable (coefficient of variation = **1.142**) — the
  model *did* learn selective attention; that selectivity simply does not align with the
  re-scoring task.
- **Comfortably edge-deployable.** Peak inference memory **70.6 MB** (14% of the 500 MB
  budget); total model size on disk **5.97 MB** (1.2% of budget); ONNX Runtime inference
  **116 FPS** (~4× the 30 FPS target).

---

## Method at a glance

The pipeline has five stages:

1. **Input image** — 640×640 RGB.
2. **YOLOv8n detector** — trained on CrowdHuman (single `person` class).
3. **Graph construction** — one node per detected box, 11 features per node, k-NN edges
   (k = 8) by box-centre Euclidean distance.
4. **GNN / GAT re-scorer** — 2-layer message passing, ≤ 38k parameters, producing a
   refined confidence in [0, 1] per box.
5. **Refined detections** — per-box re-scored confidence.

Training is **decoupled**: the detector is trained end-to-end first, its detections are
cached to disk, and the graph re-scorer is then trained on the cached detections. This
isolates each component's contribution for clean ablation and keeps the workflow
tractable on an 8 GB GPU.

Full detail is in [`docs/methodology.md`](docs/methodology.md) and
[`docs/architecture.md`](docs/architecture.md).

---

## Results

### Accuracy ablation (true/false-detection separation accuracy)

| Configuration            | Accuracy | Δ vs raw confidence | Notes                                   |
|--------------------------|:--------:|:-------------------:|-----------------------------------------|
| Raw detector confidence  |  0.7340  |          —          | Detector's own scores, thresholded @0.5 |
| GNN, 5 features          |  0.8558  |     +12.18 pp       | Graph context helps                     |
| GAT, 5 features          |  0.8498  |     +11.58 pp       | Attention slightly below uniform agg.   |
| **GNN, 11 features**     | **0.8594** |   **+12.54 pp**   | **Best configuration**                  |
| GAT, 11 features         |  0.8496  |     +11.56 pp       | Richer features do not help the GAT     |

![Accuracy ablation](figures/ablation.png)

### Efficiency

| Configuration                          |  FPS  | Notes                                   |
|----------------------------------------|:-----:|-----------------------------------------|
| PyTorch native, serial                 | 22.6  | Below 30 FPS target                     |
| PyTorch full pipeline (det+graph+GNN)  | 25.4  | Below target                            |
| ONNX Runtime, full pipeline            | 27.6  | Python pre/post-processing dominates    |
| ONNX Runtime, inference-only           | 116.1 | ~4× over target                         |

| Metric                                  | Measured | Target   | Headroom |
|-----------------------------------------|:--------:|:--------:|:--------:|
| Peak GPU memory (full pipeline)         | 70.6 MB  | < 500 MB | 86%      |
| Total model size on disk (detector+GNN) | 5.97 MB  | < 500 MB | 98.8%    |

The gap between 27.6 FPS (full pipeline) and 116.1 FPS (inference-only) shows that
~89% of per-frame end-to-end latency is **Python-side pre/post-processing**, not model
inference. Closing it (GPU-side preprocessing, CUDA-NMS) does not alter the trained model.

![Efficiency](figures/efficiency.png)

### Scene graph and attention

A scene graph on a dense CrowdHuman frame (27 detected persons, 216 k-NN edges):

![Scene graph](figures/scene_graph.png)

Attention weights from the trained GAT (averaged across heads); weight CV = 1.142
indicates selective — not near-uniform — attention:

![Attention weights](figures/attention_weights.png)

---

## Repository structure

```
aadmn/
├── README.md                  ← you are here
├── LICENSE                    ← All Rights Reserved (pre-publication)
├── CITATION.cff               ← how to cite this work
├── .gitignore
├── docs/
│   ├── methodology.md         ← detector, caching, graph, re-scorer, training
│   ├── architecture.md        ← stage-by-stage architecture + model specs
│   ├── results.md             ← full results, ablation arc, negative-result analysis
│   └── code-availability.md   ← why code is withheld + how to request it
└── figures/                   ← five publication-quality figures
```

---

## Code availability

The implementation (detector training, detection caching, graph construction, the
SceneGNN/SceneGAT re-scorers, efficiency measurement, and ONNX export) is **withheld
pending publication** to protect the contribution prior to peer review.

The code will be released under an open licence **upon publication of the associated
manuscript**. In the interim, it is available **from the author on reasonable request**
for the purposes of peer review and verification. See
[`docs/code-availability.md`](docs/code-availability.md).

This repository documents the work in full — method, configurations, hyperparameters,
results, and figures — so that the contribution is verifiable without exposing the
source for unrestricted copying.

---

## Citation

If you reference this work, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff). A BibTeX entry will be added once the manuscript is
published.

---

## License

© 2026 [Your Name]. **All rights reserved.** This repository and its contents
(documentation, figures, and results) are provided for viewing and academic reference
only. No copying, redistribution, or derivative works are permitted without prior
written consent. See [LICENSE](LICENSE).
