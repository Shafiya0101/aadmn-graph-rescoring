# Results

## 1. Accuracy ablation

"True/false-detection separation accuracy" measures, for each candidate detection, the
fraction the model correctly classifies as a real detection (IoU ≥ 0.5 with ground truth)
vs a false positive. It is related to but distinct from mAP, and it directly evaluates
the re-scoring layer's contribution.

| Configuration            | Accuracy | Δ vs baseline | Notes                                   |
|--------------------------|:--------:|:-------------:|-----------------------------------------|
| Raw detector confidence  |  0.7340  |       —       | Detector's own scores, thresholded @0.5 |
| GNN, 5 features          |  0.8558  |   +12.18 pp   | First evidence graph context helps      |
| GAT, 5 features          |  0.8498  |   +11.58 pp   | Attention slightly below uniform agg.   |
| **GNN, 11 features**     | **0.8594** | **+12.54 pp** | Best configuration                    |
| GAT, 11 features         |  0.8496  |   +11.56 pp   | Richer features do not help the GAT     |

## 2. The ablation arc

1. **5-feature GNN.** Unambiguously positive: +12.18 pp over raw confidence. Loss fell
   steadily, validation accuracy plateaued near 0.86.
2. **5-feature GAT.** Slightly *below* the GNN (+11.58 pp). Loss plateaued higher
   (0.31 vs 0.24). Two hypotheses: the GAT was under-fit (≈7× the parameters), or
   under-featured (5 features insufficient for useful selective weighting).
3. **Path B — 11 features, both models retrained.** Richer features helped the GNN
   (+12.18 → +12.54 pp) but **not** the GAT (+11.58 → +11.56 pp). Both models were
   retrained so the comparison stayed fair (an 11-feature GAT vs 5-feature GNN would be
   apples-to-oranges).

**Conclusion:** adaptive attention does not earn its place on this task.

## 3. Why attention does not help (mechanistic analysis)

The trained GAT's first-layer attention weights have **coefficient of variation = 1.142**
— high variability, not near-uniform. The model *did* learn selective attention; the
selectivity simply does not align with what distinguishes true detections from false
positives. Candidate explanations, in order of plausibility:

1. **Task–architecture mismatch.** Uniform aggregation (averaging over 8 neighbours) is a
   robust, low-variance estimator well-suited to "does the confidence pattern around this
   box look like a real-detection pattern?". Attention introduces variance that does not
   pay off here.
2. **Training-signal misalignment.** The IoU-based binary label is coarse; attention may
   learn real scene structure that is uncorrelated with the IoU label.
3. **Hyperparameter sensitivity.** The GAT was trained under the GNN's regime for fair
   comparison; per-model tuning was not performed.

This turns a vague negative ("attention failed to learn") into a sharp one: attention's
failure here is a failure of **task alignment**, not of learning.

## 4. Efficiency

| Configuration                          |  FPS  | Notes                                   |
|----------------------------------------|:-----:|-----------------------------------------|
| PyTorch native, serial                 | 22.6  | Below 30 FPS target                     |
| PyTorch full pipeline (det+graph+GNN)  | 25.4  | Below target                            |
| ONNX Runtime, full pipeline            | 27.6  | Marginally below — Python pre/post dom.  |
| ONNX Runtime, inference-only           | 116.1 | ~4× over target                         |

| Metric                                  | Measured | Target   | Headroom |
|-----------------------------------------|:--------:|:--------:|:--------:|
| Peak GPU memory (full pipeline)         | 70.6 MB  | < 500 MB | 86%      |
| Total model size on disk (detector+GNN) | 5.97 MB  | < 500 MB | 98.8%    |

A note on batching: at batch = 16 the PyTorch detector ran *slower* (16.8 FPS) than
serial, with peak memory jumping ~10× (32 → 339 MB). The likely cause is memory pressure
on an 8 GB GPU also driving the desktop display; this would not occur on a dedicated edge
device.

## 5. Limitations

- Single object class (`person`) only.
- Static spatial graphs only — no temporal/dynamic component.
- The headline metric is re-scoring separation accuracy, not mAP; a full mAP re-evaluation
  with graph-refined scores is identified as future work.
- The GAT was not given per-model hyperparameter tuning (held to the GNN regime for a fair
  comparison).
- No formal MobileNetV2/V3 backbone comparison.
- All measurements are on a single environment (one RTX A2000 8 GB laptop GPU).
- CrowdHuman-only; cross-dataset generalisation untested.

## 6. Future work (high-leverage first)

1. **Full mAP evaluation** of the GNN-refined detector (converts the +12.54 pp re-scoring
   result into a directly publishable mAP improvement).
2. **Confidence-threshold sweep / Average Precision** instead of a fixed 0.5 threshold.
3. **Temporal / dynamic graph** component (MOT17 + a tracker + temporal edges) to complete
   the original AADMN specification.
4. **Formal MobileNetV2/V3 backbone** comparison.
5. **Attention investigation** — ranking-style loss, density-aware attention, or
   end-to-end training to probe *why* selectivity does not help here.
