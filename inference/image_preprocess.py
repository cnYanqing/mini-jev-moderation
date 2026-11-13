"""Static-image normalization used by the visual review guard."""

from PIL import Image, ImageOps


def normalize(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((512, 512))
        return image
