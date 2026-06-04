# Architecture

![AADMN pipeline](../figures/architecture.png)

The AADMN pipeline is five stages. Training is decoupled: the detector is trained
end-to-end, and the graph re-scorer is trained separately on cached detections.

## Stage 1 — Input
640×640 RGB image.

## Stage 2 — YOLOv8n detector
Trained on CrowdHuman (single `person` class). Outputs candidate boxes with confidence
scores. Baseline metrics: mAP50 = 0.819, mAP50–95 = 0.515.

## Stage 3 — Graph construction
- One node per detected box.
- 11-dimensional node features (position, size, confidence, area, aspect ratio, local
  density, distance to nearest box, mean neighbour confidence, relative size).
- k-NN edges with k = 8 by box-centre distance.

## Stage 4 — GNN / GAT re-scorer
Two-layer message passing followed by a small MLP head, producing a refined confidence
in [0, 1] per box.

| Variant        | Layers              | Heads | Hidden | Embed | Activation | Params (11-feat) |
|----------------|---------------------|:-----:|:------:|:-----:|------------|:----------------:|
| SceneGNN       | 2 × SAGEConv        |   —   |   64   |  32   | ReLU       | 6,145            |
| SceneGAT       | 2 × GATConv         |   4   |   64   |  32   | ELU        | 37,185           |

Shared head: `Linear(32, 16) → ReLU → Linear(16, 1) → Sigmoid`.
Dropout 0.2 between message-passing layers in both variants.

## Stage 5 — Refined detections
Per-box re-scored confidence values.

## Why decoupled training

End-to-end training of detector + graph layer was memory-prohibitive on an 8 GB GPU.
Caching the detector's outputs once and training the graph layer on the cache:

- isolates each component's contribution (clean ablation),
- makes graph-layer iteration fast (no detector forward pass per experiment),
- fits the available hardware.

## Deployment

Exported to ONNX (opset 12, simplified, FP32) and run under ONNX Runtime with the CUDA
execution provider. Inference-only throughput is 116 FPS; the full pipeline (including
Python-side preprocessing and NMS) is 27.6 FPS, so the integration overhead — not the
model — is the bottleneck.
