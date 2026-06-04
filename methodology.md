# Methodology

This document describes the AADMN method in enough detail to understand and evaluate
the contribution. It deliberately contains **no source code**; see
[`code-availability.md`](code-availability.md).

## 1. Detector

- **Model:** YOLOv8n, initialised from COCO-pretrained weights.
- **Dataset:** CrowdHuman, converted from the native `.odgt` format (full-body boxes in
  pixel coordinates) to YOLO format (normalised centre-x/y/w/h). Single object class:
  `person`.
- **Splits:** 15,000 train images, 4,370 validation images.
- **Training:** 50 epochs, input resolution 640×640, batch size 8, mixed-precision.
  The full 50 epochs completed without early-stopping triggering (patience = 10),
  indicating room for further improvement with a longer run; this was deliberately not
  pursued to keep the baseline reproducible.
- **Result:** mAP50 = 0.8191, mAP50–95 = 0.5146; internal inference latency ≈ 2.9 ms/image.

## 2. Detection caching

The trained detector is run **once** over both splits. For each image, the detections —
bounding boxes (`xyxy`, pixel coordinates), confidence scores, class IDs, and image
dimensions — are written as one JSON record per image. This cache is the input to all
graph-layer experiments, which **decouples** detector training from re-scorer training:
each component's contribution can be isolated, and the workflow fits comfortably on an
8 GB GPU.

## 3. Graph construction

For each cached detection record a scene graph is built:

- **Nodes:** one per detected box.
- **Node features (11 dimensions):**
  1. `center_x`, 2. `center_y` — normalised box centre
  3. `width`, 4. `height` — normalised box dimensions
  5. `confidence` — detector confidence
  6. `box_area` — normalised box area
  7. `aspect_ratio` — height / width
  8. `local_density` — fraction of other boxes within ¼ of the image diagonal
  9. `dist_nearest` — distance to nearest other box, normalised by the diagonal
  10. `mean_neighbor_conf` — mean confidence of the 8 nearest neighbours
  11. `rel_size` — this box's area relative to the scene's mean box area
- **Edges:** k-nearest-neighbours with **k = 8** by box-centre Euclidean distance.
  This addresses the graph-scalability concern: a fully-connected graph over N = 100
  people has ~10,000 edges, whereas k-NN with k = 8 has ~800, while still capturing the
  local context that is relevant for re-scoring.

The 11-feature representation was **not** the original design. The first ablation used
only the first five features (position, size, confidence). Features 6–11 were added
during a follow-up experiment ("Path B") to test whether richer context would let
attention earn its place.

## 4. Re-scorer (two variants)

Both variants share an identical interface and identical training regime so the
comparison is **purely architectural**.

- **SceneGNN (GraphSAGE):** two SAGEConv layers, hidden dim 64, output embedding dim 32,
  ReLU + 0.2 dropout between layers. 6,145 parameters at 11-feature input.
- **SceneGAT (Graph Attention Network):** two GATConv layers with 4 attention heads
  (concatenated in the first layer, averaged in the second), hidden dim 64, output
  embedding dim 32, ELU + 0.2 dropout. 37,185 parameters at 11-feature input.

Both feed a small fully-connected head — `Linear(32, 16) → ReLU → Linear(16, 1) →
Sigmoid` — producing one re-scored confidence per detected box.

## 5. Training the re-scorer

- **Labels:** IoU-based matching against ground truth. A detection is labelled `1` (true)
  if its IoU with any ground-truth box ≥ 0.5, else `0` (false positive).
- **Loss:** binary cross-entropy.
- **Regime:** 15 epochs, Adam, learning rate 1e-3, batch size 16 images (variable nodes
  per image). The **same** regime is used for both models — no per-model tuning — so the
  comparison reflects architecture alone.

## 6. Deployment path

The detector is exported to ONNX (opset 12, `simplify=True`, FP32). Before timing, an
output-equivalence check compares box counts between the PyTorch and ONNX models over 20
random validation images: 18/20 matched within ±1 box, mean count difference 0.40 —
confirming the exported model reproduces the original's detections.

## Scope decisions

Three deliberate substitutions were made relative to the original brief, for tractability
on a single 8 GB laptop GPU:

1. **YOLOv8n** in place of an explicit MobileNetV2/V3 backbone (YOLOv8n is itself
   lightweight and meets the efficiency targets; a formal MobileNet comparison is future
   work).
2. **Decoupled training** of the graph layer rather than end-to-end (necessary on 8 GB,
   and a clean choice for isolating each component).
3. **Static spatial graphs** rather than temporal/dynamic graphs (the temporal extension
   requires video data and a tracker; it is documented as future work).
