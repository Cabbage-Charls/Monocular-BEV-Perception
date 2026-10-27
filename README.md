# Lightweight Monocular BEV Perception (LiDAR-Supervised Scale Estimation)

This repository contains the PyTorch implementation of a lightweight end-to-end monocular BEV perception pipeline. **Train with LiDAR, infer with Camera only.** 

## Key Results
*   **AbsRel (Hold-out OOD Test)**: 0.208 (Oracle upper bound: 0.196)
*   **MAE_S**: 1.089 m
*   **Status**: Highly consistent with validation set, proving zero overfitting and high cross-scene generalization.

## Pipeline Architecture
`Image` → `YOLOv8 (2D Detection) + Depth-Anything-V2 (Relative Depth) + SegFormer (Semantic Mask)` → `Anchor-based MLP (Scale Regression)` → `BEV Projection`

##Qualitative Results
*(在这里插入你的最终 BEV 鸟瞰图)*
<!-- 如果你在 GitHub 网页上，可以直接把本地生成的图拖拽到这两行之间，它会自动生成链接 -->

## ⚙How to Run
1. Modify the dataset paths in `config.py`.
2. Generate the cache with LiDAR supervision: `python build_cache.py`
3. Train the lightweight scale prediction MLP: `python train.py`
4. Evaluate the metrics: `python eval.py`
5. Generate BEV visualization: `python bev_inference.py`