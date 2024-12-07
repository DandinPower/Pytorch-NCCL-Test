import os
import torch
import torch.distributed as dist
from argparse import ArgumentParser

def setup(rank, world_size, device):
    os.environ['MASTER_ADDR'] = 'localhost'  # Use master node IP if multi-node
    os.environ['MASTER_PORT'] = '29526'     # Ensure this port is free
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=device)

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    setup(rank, world_size, device)

    print(f"Rank {rank}: Finished Setup")

    # Set the device for this process based on rank
    print(f'rank[{rank}] {device}')

    print(f'rank[{rank}] before barrier')
    dist.barrier()
    print(f'rank[{rank}] after barrier')

    # Create a tensor specific to each rank
    tensor = torch.ones(5, device=device) * (rank + 1)  # Tensor values depend on rank
    print(f"Rank {rank} starting with tensor: {tensor}")

    # Test Broadcast: Rank 0 broadcasts its tensor to all other ranks
    if rank == 0:
        tensor += 10  # Modify the tensor on rank 0 before broadcasting
    dist.broadcast(tensor, src=0)
    print(f"Rank {rank} after broadcast: {tensor}")

    # Test Allreduce: Sum all tensors across ranks and distribute the result to all ranks
    tensor = torch.ones(5, device=device) * (rank + 1)  # Reset tensor values
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} after all_reduce (sum): {tensor}")

    # Test Allgather: Gather tensors from all ranks into a list of tensors
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    print(f"Rank {rank} after all_gather: {gathered_tensors}")

    cleanup()
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ngpus', type=int, required=True)
    args = parser.parse_args()
    
    world_size = args.ngpus
    # Use torch.multiprocessing to spawn processes for each GPU
    torch.multiprocessing.spawn(demo_basic,
                                args=(world_size,),
                                nprocs=world_size,
                                join=True)