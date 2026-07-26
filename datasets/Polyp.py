"""Loader for the combined Kvasir and CVC-ClinicDB semi-supervised task."""

from torch.utils.data import DataLoader

from datasets.ISIC import ISICDataset, _make_semi_supervised_sampler
from datasets.tools import ResizeAndPad, collate_fn, collate_fn_soft


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = ISICDataset(cfg, cfg.datasets.Polyp.root_dir, cfg.datasets.Polyp.test_list, transform=transform)
    train = ISICDataset(
        cfg, cfg.datasets.Polyp.root_dir, cfg.datasets.Polyp.train_list,
        transform=transform, training=True,
    )
    sampler = _make_semi_supervised_sampler(train, cfg)
    return (
        DataLoader(train, batch_size=cfg.batch_size, shuffle=sampler is None,
                   sampler=sampler, num_workers=cfg.num_workers, collate_fn=collate_fn),
        DataLoader(val, batch_size=cfg.val_batchsize, shuffle=False,
                   num_workers=cfg.num_workers, collate_fn=collate_fn),
    )


def load_datasets_soft(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = ISICDataset(cfg, cfg.datasets.Polyp.root_dir, cfg.datasets.Polyp.test_list, transform=transform)
    train = ISICDataset(
        cfg, cfg.datasets.Polyp.root_dir, cfg.datasets.Polyp.train_list,
        transform=transform, training=True, if_self_training=True,
    )
    sampler = _make_semi_supervised_sampler(train, cfg)
    return (
        DataLoader(train, batch_size=cfg.batch_size, shuffle=sampler is None,
                   sampler=sampler, num_workers=cfg.num_workers, collate_fn=collate_fn_soft),
        DataLoader(val, batch_size=cfg.val_batchsize, shuffle=False,
                   num_workers=cfg.num_workers, collate_fn=collate_fn),
    )
