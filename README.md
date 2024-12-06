# PyTorch NCCL Test

This repository is used to verify if your CUDA devices can communicate correctly in PyTorch using the NCCL library.

## Prerequisites

1.	Ensure that the NVIDIA driver is installed for your device.
2.	Check your device’s connection information:

    `nvidia-smi topo -m`

## Running the Test

1.	Modify run.sh to set the desired number of GPUs to test:

    `NGPUS=<number_of_gpus>`


2.	Run the script:

    `bash run.sh`

## Additional Tests

If you are interested in more tests, such as bandwidth testing, you can use the following repository:

1. [NVIDIA NCCL Tests](https://github.com/NVIDIA/nccl-tests)