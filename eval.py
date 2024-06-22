"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import argparse
from pathlib import Path

import torch

from utils import refine, save_tensor_to_image
from datasets import get_test_loader
from models import Generator
from sconf import Config
from train import setup_transforms
import datetime


def eval_ckpt():
    start_time = datetime.datetime.now(datetime.UTC)

    parser = argparse.ArgumentParser()
    parser.add_argument("config_paths", nargs="+", help="path to config.yaml")
    parser.add_argument("--weight", help="path to weight to evaluate.pth")
    parser.add_argument("--result_dir", help="path to save the result file")
    args, left_argv = parser.parse_known_args()

    cfg = Config(*args.config_paths, default="cfgs/defaults.yaml")
    cfg.argv_update(left_argv)
    img_dir = Path(args.result_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    trn_transform, val_transform = setup_transforms(cfg)

    g_kwargs = cfg.get('g_args', {})
    gen = Generator(1, cfg.C, 1, **g_kwargs).cuda()

    weight = torch.load(args.weight)
    if "generator_ema" in weight:
        weight = weight["generator_ema"]
    gen.load_state_dict(weight)
    test_dset, test_loader = get_test_loader(cfg, val_transform)
    generated = 0
    for batch in test_loader:
        style_imgs = batch["style_imgs"].cuda()
        char_imgs = batch["source_imgs"].unsqueeze(1).cuda()
        out = gen.gen_from_style_char(style_imgs, char_imgs)
        fonts = batch["fonts"]
        chars = batch["chars"]
        print(f"generate batch: {chars}")
        for image, font, char in zip(refine(out), fonts, chars):
            (img_dir / font).mkdir(parents=True, exist_ok=True)
            path = img_dir / font / f"{char}.png"  # TODO: 改成輸出unicode編號
            save_tensor_to_image(image, path)
            generated += 1
    finish_time = datetime.datetime.now(datetime.UTC)
    time_spent = (finish_time - start_time).total_seconds()
    days = time_spent // 86400
    hours = time_spent % 86400 // 3600
    minutes = time_spent % 3600 // 60
    seconds = time_spent % 60
    print(f"總共耗時{days}天{hours}小時{minutes}分{seconds}秒，總共造出{generated}個中文字")


if __name__ == "__main__":
    eval_ckpt()
