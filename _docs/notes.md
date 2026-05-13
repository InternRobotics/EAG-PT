# EAG-PT Notes

## Machine Specification

We use a Linux server with RTX 4090 to run all our experiments. The system image we choose is [nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/12.8.1-cudnn-devel-ubuntu22.04/images/sha256-61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719). Versions are shown below.

```sh
nvidia-smi
# NVIDIA GeForce RTX 4090
# NVIDIA-SMI 570.153.02
# Driver Version: 570.153.02
# CUDA Version: 12.8

nvcc -V
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2025 NVIDIA Corporation
# Built on Fri_Feb_21_20:23:50_PST_2025
# Cuda compilation tools, release 12.8, V12.8.93
# Build cuda_12.8.r12.8/compiler.35583870_0

c++ --version
# c++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

g++ --version
# g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

apt list --installed | grep libnvidia-gl
# libnvidia-gl-570/unknown,now 570.211.01-0ubuntu1 amd64 [installed]
```

## About opencv-python

You can just use the package `opencv-python` instead of `opencv-python-headless` with the same version. On our machine, these libraries should be installed:

```sh
apt install libgl-dev
# should fix: ImportError: libGL.so.1: cannot open shared object file: No such file or directory
apt install libglib2.0-0
# should fix: ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory
```
