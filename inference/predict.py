"""Run the accepted Mini-Jev image v6 weights from an unpacked release bundle."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "1")

import torch
from PIL import Image, ImageOps
from jev.decide import Decider
from routed_head import RoutedHead
from patch_embed import install_patch_embedding
from diagram_guard import is_monochrome, score as diagram_score
from overnight_rules import decision


ROOT = Path(__file__).resolve().parent
HEADS = ("text_sexual", "text_violence", "image_sexual", "image_violence")


def load(device=None):
    policy = json.loads((ROOT / "policy.json").read_text(encoding="utf-8"))
    chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if chosen == "cuda" else torch.float32
    model = Decider.from_pretrained(
        str(ROOT), device=chosen, dtype=dtype, max_pixels=262144, cache_prefix=True
    )
    install_patch_embedding(model.model)
    states = {
        name: torch.load(ROOT / "heads" / f"{name}.pt", map_location="cpu", weights_only=True)
        for name in HEADS
    }
    model.model.head = RoutedHead(states, policy["hidden"], policy["head_dim"]).to(chosen).eval()
    model.model.head.modality = "image"
    return model, policy, chosen


def score_image(path, model, policy, device):
    with Image.open(path) as opened:
        if getattr(opened, "n_frames", 1) > 1:
            raise ValueError("Animated or multi-page images require frame sampling")
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((512, 512))
    try:
        answers = model.decide(policy["prompt"], policy["questions"], images=[image])["answers"]
    finally:
        image.close()
    scores = {
        "sexual": float(answers["sexual"]["probabilities"]["yes"]),
        "gore": float(answers["violence"]["probabilities"]["yes"]),
    }
    guard = policy.get("diagram_guard")
    dg = diagram_score(path, guard) if guard else None
    mono = is_monochrome(path) if policy.get("monochrome_gore_review") else False
    suggestion = decision(
        scores["sexual"], scores["gore"], policy["block_thresholds"],
        policy["review_thresholds"], dg,
        guard["threshold"] if guard else 1.01, mono,
    )
    return {
        "scores": scores,
        "suggested_action": suggestion,
        "decision": "review",
        "mode": "review_only",
        "device": device,
        "probability_calibrated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    model, policy, device = load(args.device)
    print(json.dumps(score_image(args.image, model, policy, device), ensure_ascii=False))


if __name__ == "__main__":
    main()
