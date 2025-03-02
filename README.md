<p align="center">

  <h2 align="center">Synthesizing Physically Plausible Human Motions in 3D Scenes</h2>
  <p align="center">
    <a href="https://liangpan99.github.io/"><strong>Liang Pan</strong></a>
    ·  
    <a href="https://scholar.google.com/citations?user=GStTsxAAAAAJ&hl=en&oi=ao"><strong>Jingbo Wang</strong></a>
    ·
    <a href="http://www.buzhenhuang.com/"><strong>Buzhen Huang</strong></a>
    ·
    <a href="https://budiu-39.github.io/"><strong>Junyu Zhang</strong>
    ·
    <a href="https://haofanwang.github.io/"><strong>Haofan Wang</strong></a>
    ·
    <a href="https://tangxuvis.github.io/"><strong>Xu Tang</strong></a>
    ·
    <a href="https://www.yangangwang.com/"><strong>Yangang Wang</strong></a>
    <br>
    Southeast University&emsp;Shanghai AI Laboratory&emsp;Xiaohongshu Inc.
  </p>
  <h2 align="center">3DV 2024</h2>
  <img src='https://github.com/liangpan99/InterScene/blob/main/docs/assets/teaser.png'>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2308.09036">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
    <a href='https://liangpan99.github.io/InterScene'>
      <img src='https://img.shields.io/badge/InterScene-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=D35400' alt='Project Page'></a>
  </p>

We propose InterScene, a novel method that generates physically plausible long-term motion sequences in 3D indoor scenes. Our approach enables physics-based characters to exhibit natural interaction-involved behaviors, such as sitting down (gray), getting up (blue), and walking while avoiding obstacles (pink).

## News
- **[2025-03-03]** Update camera-ready paper and website.
- **[2023-11-09]** Release code for training and evaluating the sit policy.
- **[2023-10-16]** Paper got accepted by 3DV 2024.

## Dependencies

### Environment

To create the environment, follow the following instructions: 

1. We recommend to install all the requirements through Conda by
```
conda create -n rlgpu python=3.7
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

2. Download IsaacGym Preview 4 from the [official site](https://developer.nvidia.com/isaac-gym) and install it via pip.


### Dataset

To prepare data for training/evaluating InterCon (sit & get-up policies), follow the following instructions:

1. Download SMPL-X v1.1 from the [official site](https://smpl-x.is.tue.mpg.de/). Put them in the `body_models/smplx` folder. 

2. Download SAMP motion dataset from the [official site](https://samp.is.tue.mpg.de/). Put them in the `samp` folder. Please download Motion Clips (.pkl), which contains the SMPL-X parameters.

3. The file structure should look like this:

```
|-- InterScene
|-- body_models
    |-- smplx
        |-- SMPLX_FEMALE.npz
        |-- SMPLX_FEMALE.pkl
        |-- SMPLX_MALE.npz
        |-- ...
|-- samp
    |-- chair_mo_stageII.pkl
    |-- chair_mo001_stageII.pkl
    |-- chair_mo002_stageII.pkl
    |-- ...
```

4. Run the following script to generate reference motion dataset:

```
python InterScene/data/dataset_samp_sit/generate_motion.py --samp_pkl_dir ./samp --smplx_dir ./body_models/smplx
```

5. Run the following script to generate 3D object dataset:

```
python InterScene/data/dataset_samp_sit/generate_obj.py
```

## Getting Started

### InterCon (sit & get-up policies)

```
## training sit policy
python InterScene/run.py --task HumanoidLocationSit --cfg_env InterScene/data/cfg/humanoid_location_sit.yaml --cfg_train InterScene/data/cfg/train/rlg/amp_task_location_sit.yaml --motion_file InterScene/data/dataset_samp_sit/dataset_samp_sit.yaml --num_envs 4096 --headless

## evaluating sit policy
python InterScene/run.py --task HumanoidLocationSit --cfg_env InterScene/data/cfg/humanoid_location_sit.yaml --cfg_train InterScene/data/cfg/train/rlg/amp_task_location_sit.yaml --motion_file InterScene/data/dataset_samp_sit/dataset_samp_sit.yaml --num_envs 4096 --headless --checkpoint InterScene/data/models/policy_sit.pth --test
```

## Citation

```bibtex
@inproceedings{pan2024synthesizing,
  title={Synthesizing physically plausible human motions in 3d scenes},
  author={Pan, Liang and Wang, Jingbo and Huang, Buzhen and Zhang, Junyu and Wang, Haofan and Tang, Xu and Wang, Yangang},
  booktitle={2024 International Conference on 3D Vision (3DV)},
  pages={1498--1507},
  year={2024},
  organization={IEEE}
}
```

## References
This repository is built on the top of the following amazing codebases: 
* Main code framework is from: [ASE](https://github.com/nv-tlabs/ASE)
* Some scripts are from: [Pacer](https://github.com/nv-tlabs/pacer/tree/main), [HuMoR](https://github.com/davrempe/humor)

Please follow the license of the above repositories for the usage of this project.