import argparse
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np


def rgb2gray(rgb):
    gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    return gray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir")
    args = parser.parse_args()

    images = list(Path(args.image_dir).rglob("*.png"))
    for image in tqdm(images):
        filename = image.stem
        dirname = image.parent
        color_image = Image.open(str(dirname / (filename + ".png")))
        cimg_arr = np.array(color_image)
        gray_arr = rgb2gray(cimg_arr).astype(np.uint8)
        gray_image = Image.fromarray(gray_arr)
        gray_image.save(f"{args.image_dir}/{filename}.png")


if __name__ == "__main__":
    main()
