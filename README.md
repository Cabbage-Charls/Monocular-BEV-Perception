# Lightweight Monocular BEV Perception (LiDAR-Supervised Scale Estimation)

This repository contains the PyTorch implementation of a lightweight end-to-end monocular BEV perception pipeline. **Train with LiDAR, infer with Camera only.** 

## Key Results
*   **AbsRel (Hold-out OOD Test)**: 0.208 (Oracle upper bound: 0.196)
*   **MAE_S**: 1.089 m
*   **Status**: Highly consistent with validation set, proving zero overfitting and high cross-scene generalization.

## Pipeline Architecture
`Image` → `YOLOv8 (2D Detection) + Depth-Anything-V2 (Relative Depth) + SegFormer (Semantic Mask)` → `Anchor-based MLP (Scale Regression)` → `BEV Projection`

##Qualitative Results
<img width="2484" height="975" alt="bev_000013" src="https://github.com/user-attachments/assets/ff00d308-b541-4163-8132-9d7658bbd09c" />
<img width="2448" height="970" alt="bev_000014" src="https://github.com/user-attachments/assets/6b2b9491-5bb3-4891-ad5c-fe16f0907f4b" />

<!-- 如果你在 GitHub 网页上，可以直接把本地生成的图拖拽到这两行之间，它会自动生成链接 -->

## ⚙How to Run
1. Modify the dataset paths in `config.py`.
2. Generate the cache with LiDAR supervision: `python build_cache.py`
3. Train the lightweight scale prediction MLP: `python train.py`
4. Evaluate the metrics: `python eval.py`
5. Generate BEV visualization: `python bev_inference.py`
