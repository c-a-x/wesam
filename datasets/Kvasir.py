"""Kvasir loader for the IFP target_images/target_masks layout."""

from torch.utils.data import DataLoader

from datasets.ISIC import ISICDataset, _make_semi_supervised_sampler
from datasets.tools import ResizeAndPad, collate_fn, collate_fn_soft


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = ISICDataset(cfg, cfg.datasets.Kvasir.root_dir, cfg.datasets.Kvasir.test_list, transform=transform)
    train = ISICDataset(
        cfg, cfg.datasets.Kvasir.root_dir, cfg.datasets.Kvasir.train_list, transform=transform, training=True
    )
    train_sampler = _make_semi_supervised_sampler(train, cfg)
    return (
        DataLoader(train, batch_size=cfg.batch_size, shuffle=train_sampler is None,
                   sampler=train_sampler, num_workers=cfg.num_workers, collate_fn=collate_fn),
        DataLoader(val, batch_size=cfg.val_batchsize, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn),
    )


def load_datasets_soft(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = ISICDataset(cfg, cfg.datasets.Kvasir.root_dir, cfg.datasets.Kvasir.test_list, transform=transform)
    train = ISICDataset(
        cfg, cfg.datasets.Kvasir.root_dir, cfg.datasets.Kvasir.train_list,
        transform=transform, training=True, if_self_training=True,
    )
    train_sampler = _make_semi_supervised_sampler(train, cfg)
    return (
        DataLoader(train, batch_size=cfg.batch_size, shuffle=train_sampler is None,
                   sampler=train_sampler, num_workers=cfg.num_workers,
                   collate_fn=collate_fn_soft),
        DataLoader(val, batch_size=cfg.val_batchsize, shuffle=False, num_workers=cfg.num_workers,
                   collate_fn=collate_fn),
    )
