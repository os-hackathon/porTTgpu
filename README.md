# porTTgpu
Copyright 2021 Fluid Numerics LLC

This repository is used to define HIP kernels for 3-D array transpose operations.

## Motivation
This repository is motivated by the following needs :
* GPU developers are increasingly interested in using open-source and portable ("code once, run anywhere") software for GPU acceleration.
* 3-D Array Transposes are a critical operation for multi-dimensional Fourier Spectral methods and machine learning algorithms.


## Goals
Currently, porTTgpu implements "naive" tensor transpose operations that achieve suboptimal performance. 
[You can review the current benchmarks across Nvidia and AMD hardware here](https://docs.google.com/spreadsheets/d/1RQlCpgZYfRKr22wLNaT5Tbw38KD58SJSt5Yl68uD7Fs)

Our goal is to improve performance of the 3-D array transpose kernels by exploiting coallesced global memory access patterns (Contributions welcome).
