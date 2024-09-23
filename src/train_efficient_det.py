import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Adam

from dataset.anchor_based_dataset import AnchorBasedDataset, anchor_based_dataset_collate_fn
from models.simplified_efficient_det import SimplifiedEfficientDet, EfficientDetLossFn
from utils.constants import *
from utils.config import EFFICIENT_DET_WEIGHTS_PATH, SPLIT_TRAIN, SPLIT_VAL


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def initialize_model(device):
    model = SimplifiedEfficientDet(N_LABELS, N_ANCHORS_PER_CELL).to(device)
    if os.path.exists(EFFICIENT_DET_WEIGHTS_PATH):
        model.load_state_dict(torch.load(EFFICIENT_DET_WEIGHTS_PATH, map_location=device))
    # Wrap model in DDP
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    return model


def save_model(model, rank):
    if rank == 0:  # Save only from rank 0 (main process)
        os.makedirs(os.path.dirname(EFFICIENT_DET_WEIGHTS_PATH), exist_ok=True)
        torch.save(model.module.state_dict(), EFFICIENT_DET_WEIGHTS_PATH)


def initialize_data_loaders(device, world_size, rank):
    data_loaders = {}
    for split in [SPLIT_TRAIN, SPLIT_VAL]:
        dataset = AnchorBasedDataset(split, device)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                                collate_fn=anchor_based_dataset_collate_fn)
        data_loaders[split] = dataloader
    return data_loaders


def train(rank, world_size, model, dataloaders):
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    optimizer = Adam(model.parameters(), LEARNING_RATE)
    loss_fn = EfficientDetLossFn()

    best_loss = float("inf")
    for epoch in range(N_EPOCH + 1):
        for split in [SPLIT_TRAIN, SPLIT_VAL]:
            dataloader = dataloaders[split]
            if split == SPLIT_TRAIN:
                model.train()
            else:
                model.eval()

            total_loss = 0
            for batch_idx, (images, tgt_cls, tgt_bbox) in enumerate(dataloader):
                images = images.to(device)
                tgt_cls = tgt_cls.to(device)
                tgt_bbox = tgt_bbox.to(device)

                optimizer.zero_grad()
                src_cls, src_bbox = model(images)
                batch_loss = loss_fn(src_cls, src_bbox, tgt_cls, tgt_bbox)
                total_loss += batch_loss.item()

                if split == SPLIT_TRAIN:
                    batch_loss.backward()
                    optimizer.step()

                # Only rank 0 prints progress
                if rank == 0:
                    print(
                        f"\rEpoch {epoch}, {split}, loss:{batch_loss:.5f}, "
                        f"progress: {(batch_idx + 1) / len(dataloader) * 100:.2f}%",
                        end='           ',
                        flush=True
                    )

            avg_loss = total_loss / len(dataloader)

            if rank == 0:
                print(f"\nEpoch {epoch}, {split} split, Avg Loss: {avg_loss:.5f}")

            if split == SPLIT_VAL and avg_loss < best_loss and rank == 0:
                best_loss = avg_loss
                save_model(model, rank)

    cleanup_ddp()

## TODO test this last code, because it is first time using DDP, if you want to back you can write git reset --hard
## TODO Run code in background

if __name__ == "__main__":
    world_size =  torch.cuda.device_count()
    device = torch.device(f'cuda:0')

    # Initialize model and data loaders outside of train function
    model = initialize_model(device)
    dataloaders = initialize_data_loaders(device, world_size, rank=0)

    # Spawn the training process with model and data loaders passed in
    torch.multiprocessing.spawn(train, args=(world_size, model, dataloaders), nprocs=world_size, join=True)
