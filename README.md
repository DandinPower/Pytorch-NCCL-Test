# Pytorch NCCL Test

This repository is used to check your cuda devices can communicate fine in PyTorch by using nccl library.

## Prerequisite

1. Please ensure you are installed nvidia driver for your device
2. Check your device connection information
    ```bash
    nvidia-smi topo -m
    ```

## Run the test

1. modify `run.sh` to set the correct GPU number you want to test
    ```NGPUS=<the number of gpu>```
2. run the script
    ```bash
    bash run.sh
    ```


## Other Test

if you are interesting in more test like bandwidth, you can use following repository:

1. https://github.com/NVIDIA/nccl-tests