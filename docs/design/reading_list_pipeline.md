# Reading list for the Duet capture-and-processing pipeline

Compiled 2026-09-23. Every link was checked; nothing here is cited from memory. Grouped by the stage of the pipeline it informs. Read the starred (★) items first.

## 1. Head tracking (SLAM / visual-inertial odometry)

Classic, run-anywhere, well understood. Use these as the reference implementations and as the "offline SLAM from cheap video + IMU" path.
- ★ ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM (Campos et al., 2020). https://arxiv.org/abs/2007.11898 · code https://github.com/UZ-SLAMLab/ORB_SLAM3
- VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator (Qin, Li, Shen, 2017). https://arxiv.org/abs/1708.03852 · code https://github.com/HKUST-Aerial-Robotics/VINS-Mono
- OpenVINS (filter-based VIO, MSCKF). docs https://docs.openvins.com/ · code https://github.com/rpng/open_vins

Learned SLAM. More robust to blur, textureless walls and rolling shutter; what we would run offline on server GPUs for cheap headbands.
- ★ DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras (Teed, Deng, 2021). https://arxiv.org/abs/2108.10869
- Deep Patch Visual Odometry (DPVO) (Teed, Lipson, Deng, NeurIPS 2023). https://arxiv.org/abs/2208.04726 · Deep Patch Visual SLAM https://arxiv.org/pdf/2408.01654
- ★ MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors (Murai, Dexheimer, Davison, CVPR 2025). https://arxiv.org/abs/2412.12392
- VGGT-SLAM 2.0: Real-time Dense Feed-forward Scene Reconstruction. https://arxiv.org/pdf/2601.19887

## 2. Feed-forward 3D and 4D reconstruction (geometry without a calibration rig)

- DUSt3R: Geometric 3D Vision Made Easy (Wang et al., CVPR 2024). https://arxiv.org/abs/2312.14132 · code https://github.com/naver/dust3r
- ★ VGGT: Visual Geometry Grounded Transformer (CVPR 2025 best paper). https://arxiv.org/abs/2503.11651 · code https://github.com/facebookresearch/vggt
- ★ MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion (dynamic scenes, per-timestep pointmaps). https://arxiv.org/abs/2410.03825
- Shape of Motion: 4D Reconstruction from a Single Video (ICCV 2025). https://arxiv.org/abs/2407.13764 · https://shape-of-motion.github.io/
- MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds (CVPR 2025, Penn / Daniilidis lab). https://arxiv.org/abs/2405.17421 · code https://github.com/JiahuiLei/MoSca
- PAGE-4D and GEM-4D (Kaichen Zhou; separates camera motion from moving geometry; 4D correspondence for robot trajectories). https://arxiv.org/pdf/2510.17568 · https://arxiv.org/pdf/2605.22882

## 3. Two people in one frame (collaborative / multi-agent SLAM)

This is the problem CoMind solved with Meta's Multi-SLAM and we solve with a shared board at the start of each block. These papers are how to do it without the board.
- ★ Project Aria Multi-SLAM output format (what CoMind ships; our `shared_world_features.py` consumes it). https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_multi_slam
- Kimera-Multi: Robust, Distributed, Dense Metric-Semantic SLAM for Multi-Robot Systems (2021). https://arxiv.org/abs/2106.14386
- CCM-SLAM: centralized collaborative monocular SLAM (Schmuck, Chli, 2019). code https://github.com/VIS4ROB-lab/ccm_slam
- Swarm-SLAM: Sparse Decentralized Collaborative SLAM (2023). https://arxiv.org/abs/2301.06230 · code https://github.com/MISTLab/Swarm-SLAM
- CoMo3R-SLAM: collaborative monocular dense SLAM with learned 3D priors (2026). https://arxiv.org/pdf/2605.30488

## 4. Time sync and calibration (the unglamorous part that decides data quality)

- ★ Kalibr: Unified temporal and spatial calibration for multi-sensor systems (Furgale, Rehder, Siegwart, IROS 2013). code https://github.com/ethz-asl/kalibr (camera intrinsics, camera-IMU extrinsics, time offset; use it on every ZED/Gemini rig once)
- TICSync: Knowing when things happened (Harrison, Newman, ICRA 2011): the clock-mapping algorithm Aria uses across devices. https://ieeexplore.ieee.org/document/5980112/ · Aria's write-up https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk/concepts/about_ticsync
- Project Aria coordinate conventions (device / CPF / camera frames; the source of our gaze-frame bug). https://facebookresearch.github.io/projectaria_tools/docs/data_formats/coordinate_convention/3d_coordinate_frame_convention

## 5. Hands in 3D from video (what replaces Aria's built-in hand tracking on ZED data)

- ★ HaMeR: Reconstructing Hands in 3D with Transformers (CVPR 2024). https://arxiv.org/abs/2312.05251
- ★ WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild (CVPR 2025; 130 fps, detector + reconstructor). https://arxiv.org/abs/2409.12259 · code https://github.com/rolpotamias/WiLoR
- ★ HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos (CVPR 2025; hands in the world frame + egocentric SLAM, exactly our problem). https://arxiv.org/abs/2501.02973 · code https://github.com/ThunderVVV/HaWoR

## 6. Body from a head-mounted device (Kaichen's "whole articulated body")

- ★ EgoAllo: Estimating Body and Hand Motion in an Ego-sensed World (2024; body + hands from head pose and ego images). https://arxiv.org/abs/2410.03665 · https://egoallo.github.io/
- EgoEgo: Ego-Body Pose Estimation via Ego-Head Pose Estimation (CVPR 2023). https://arxiv.org/abs/2212.04636
- EgoHumans: An Egocentric 3D Multi-Human Benchmark (ICCV 2023; other people's 3D pose from an ego view). https://arxiv.org/abs/2305.16487

## 7. Objects: identity, 6-DoF pose, and tracking (the missing ingredient for handover anticipation)

- ★ FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects (NVIDIA, CVPR 2024; works from a CAD model or a few reference images). https://arxiv.org/abs/2312.08344
- BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects (CVPR 2023; RGB-D, no model needed). https://arxiv.org/abs/2303.14158 · code https://github.com/NVlabs/BundleSDF
- CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos. https://arxiv.org/abs/2410.11831 · code https://github.com/facebookresearch/co-tracker
- HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos (CVPR 2025; the Aria hand+object ground-truth dataset and its capture method). https://arxiv.org/abs/2411.19167 · https://facebookresearch.github.io/hot3d/

## 8. Datasets to compare against and borrow protocol from

- CoMind (ETH; what all our results are on). https://comind.ethz.ch/ · paper https://arxiv.org/html/2607.06691
- ★ Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives (1,286 h; ego + exo, gaze, poses, IMU). https://arxiv.org/abs/2311.18259 · Aria docs https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/ego-exo4d
- Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild (300 h; full-body ground truth with Aria + wristbands). https://arxiv.org/abs/2406.09905
- HOT3D (above).

## 9. Handovers and human-to-robot transfer (the downstream use)

- ★ Object Handovers: a Review for Robotics (Ortenzi et al., 2020). https://arxiv.org/abs/2007.12952
- A Study of Human-Robot Handover through Human-Human Object Transfer (2023; measures human-human handovers to inform robot handovers, close to our metrics). https://arxiv.org/pdf/2311.13021
- Human-robot object handover: recent progress and future direction (2024 survey). https://www.sciencedirect.com/science/article/pii/S2667379724000032
- ★ EMMA: Scaling Mobile Manipulation via Egocentric Human Data (Lawrence's paper; human ego data + static robot data, Handover Wine task). https://arxiv.org/abs/2509.04443
- Not verified: Lawrence recommended a paper he called "COALESCE" on handover policies conditioned on the receiver. I could not find it under that name; ask him for the exact title before citing it.

## Suggested reading order for the team

1. Aria coordinate conventions + TICSync page (one hour; prevents the two frame bugs we already hit).
2. ORB-SLAM3 and DROID-SLAM (how head tracking works, classic vs learned).
3. HaWoR, then WiLoR (hands in the world frame from ego video; the ZED replacement for Aria hand tracking).
4. FoundationPose (object pose from tags today, from a model tomorrow).
5. MASt3R-SLAM and MonST3R (where dense/dynamic reconstruction is going; what "4D" means in practice).
6. Kimera-Multi (how two maps become one without a board).
7. Ego-Exo4D and HOT3D papers, capture-protocol sections only.
8. The handover review and EMMA (what the data is for).
