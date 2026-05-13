# EAG-PT: Emission-Aware Gaussians and Path Tracing for Diffuse Indoor Scene Reconstruction and Editing

[**Project Page**](https://eag-pt.github.io) | [**Paper**](https://arxiv.org/abs/2601.23065) | [**EAG-PT-tracer**](https://github.com/InternRobotics/EAG-PT-tracer)

[Xijie Yang](https://xijie-yang.github.io), [Mulin Yu](https://mulinyu.github.io/)\*, [Changjian Jiang](https://scholar.google.com/citations?user=V4miywEAAAAJ), [Kerui Ren](https://cskrren.github.io/), [Tao Lu](https://inspirelt.github.io/), [Jiangmiao Pang](https://oceanpang.github.io/), [Dahua Lin](https://www.ie.cuhk.edu.hk/faculty/lin-dahua/), [Bo Dai](https://datascience.hku.hk/people/bo-dai/)\*, [Linning Xu](https://eveneveno.github.io/lnxu/)

![](./_media/Method-pipeline.jpg)

## Get Started

### Download Code

```sh
git clone https://github.com/InternRobotics/EAG-PT.git
git clone https://github.com/InternRobotics/EAG-PT-tracer.git
```

### Prepare Data

Please check [`_data/README.md`](_data/README.md) to download multi-view indoor scenes used in the paper.

### Setup Environment

We tested the code on Ubuntu 22.04 with NVIDIA RTX 4090, Driver 570.153.02, CUDA 12.8, nvcc 12.8, and OptiX 7.7. For machine specification, please check [`_docs/notes.md`](_docs/notes.md). The code should also work on other NVIDIA RTX cards with higher CUDA versions.

```sh
# (optionally) This adds /lib/x86_64-linux-gnu/libnvoptix.so.1 and /usr/share/nvidia/nvoptix.bin.
apt install libgl-dev libnvidia-gl-570
```

```sh
conda create -y -n EAG-PT python=3.11
conda activate EAG-PT

pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==2.2.6 einops==0.8.1 tqdm==4.67.1 opencv-python-headless==4.12.0.88 OpenImageIO==3.0.4.0 mitsuba==3.7.1 plyfile==1.1.3 open3d==0.19.0 lpips==0.1.4 flip-evaluator==1.7
```

### Install Tracer

```sh
cd EAG-PT-tracer/

pip install ninja==1.13.0
rm -rf *_optix/ptx/ build/ *.egg-info/; TORCH_CUDA_ARCH_LIST=8.9 pip install . -v --no-build-isolation
```

- arch 8.9 is for RTX 4090, for other gpus, check https://developer.nvidia.com/cuda-gpus. arch should also be changed in `setup.py` `command_nvcc` (and in `eag_pt_tracer_optix/program/generate-ptx.sh`).
- If the tracer cannot be installed and used successfully, please open an issue in [EAG-PT-tracer](https://github.com/InternRobotics/EAG-PT-tracer).

### Run Code

```sh
cd EAG-PT/

# Reconstruct the scene.
_scripts/0-and-1-reconstruction.sh
# Edit and render the scene.
_scripts/editing-and-rendering.sh
# Bake light on edited scene.
_scripts/light-baking.sh
```

- Scenes, scenarios, parameters could be manually changed in the script.
- Scene editing may require some assets from other reconstructed scenes.
- Results will be saved in [`_output/`](_output/).
- For viewing EXR files, use the VS Code extension [`VERIV`](https://marketplace.visualstudio.com/items?itemName=mcrespo.veriv) or local software [`tev`](https://github.com/Tom94/tev).

## Work in Progress

- [ ] Upload EFT data.
- [ ] Add tracer usage examples.
- [ ] Release math details in the tracer.
- [ ] Tidy up data capture process.

## Feedback and Contributions

Issues, pull requests, and suggestions are welcome. If you find EAG-PT useful, please consider starring this repository and citing our paper 😉.

## Citation

```latex
@misc{EAG-PT-2026-XijieYang,
    title={EAG-PT: Emission-Aware Gaussians and Path Tracing for Diffuse Indoor Scene Reconstruction and Editing},
    author={Xijie Yang and Mulin Yu and Changjian Jiang and Kerui Ren and Tao Lu and Jiangmiao Pang and Dahua Lin and Bo Dai and Linning Xu},
    year={2026},
    eprint={2601.23065},
    archivePrefix={arXiv},
    primaryClass={cs.GR},
    doi={https://doi.org/10.1145/3799902.3811054},
    url={https://arxiv.org/abs/2601.23065},
}
```
