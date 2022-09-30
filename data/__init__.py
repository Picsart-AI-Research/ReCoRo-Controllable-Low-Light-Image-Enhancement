from torch.utils.data import DataLoader

from .datasets import (
    CombinedDataset,
    PairedMaskDataset,
    UnpairedMaskDataset,
    MaskDataset,
    PairedAugMaskDataset
)
from .alpha_encoder import AlphaEncoder


def get_dataset(cfg, alpha_encoder):
    if cfg.type == "mask":
        dataset = MaskDataset(cfg, alpha_encoder=alpha_encoder)
        return dataset
    elif cfg.type == "unpaired_mask":
        dataset = UnpairedMaskDataset(cfg, alpha_encoder=alpha_encoder)
        return dataset
    elif cfg.type == "paired_mask":
        dataset = PairedMaskDataset(cfg, alpha_encoder=alpha_encoder)
        return dataset
    elif cfg.type == "paired_aug_mask":
        dataset = PairedAugMaskDataset(cfg, alpha_encoder=alpha_encoder)
        return dataset



def get_data(cfg):
    if cfg.alpha_encoder is None:
        alpha_encoder = None
    else:
        alpha_encoder = AlphaEncoder(cfg.alpha_encoder)

    if cfg.type == "combined":
        dataset_A = get_dataset(cfg.dataset_A, alpha_encoder)
        dataset_B = get_dataset(cfg.dataset_B, alpha_encoder)
        dataset = CombinedDataset(cfg, dataset_A, dataset_B)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.dataloader.batch_size,
            shuffle=cfg.dataloader.shuffle,
            num_workers=cfg.dataloader.num_workers,
        )
        eval_dataset = get_dataset(cfg.eval_dataset, alpha_encoder)
        eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=cfg.dataloader.num_workers,
            )
        return dataset, dataloader, eval_dataset, eval_dataloader
    else:
        dataset = get_dataset(cfg, alpha_encoder)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.dataloader.batch_size,
            shuffle=cfg.dataloader.shuffle,
            num_workers=cfg.dataloader.num_workers,
        )
        return dataset, dataloader
