"""Near-realistic synthetic conveyor-part generator.

Produces matte/textured rectangular parts with pose, illumination, noise,
legitimate markings, and exact masks for scratch, edge and corner defects.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Sample:
    """One generated inspection sample."""
    image: np.ndarray
    mask: np.ndarray
    class_name: str
    bbox_xyxy: tuple[int, int, int, int] | None
    metadata: dict[str, Any]


def _texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Create low-frequency matte texture."""
    n = rng.normal(0, 1, (h, w)).astype(np.float32)
    s = cv2.GaussianBlur(n, (0, 0), 10)
    return s / max(float(s.std()), 1e-6)


def _scratch(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> None:
    """Render a dark scratch and exact mask in-place."""
    h, w = img.shape
    length = int(rng.integers(max(15, w // 12), max(20, w // 3)))
    x = int(rng.integers(w // 4, 3 * w // 4))
    y = int(rng.integers(h // 4, 3 * h // 4))
    a = float(rng.uniform(0, np.pi))
    dx = int(length * np.cos(a) / 2)
    dy = int(length * np.sin(a) / 2)
    width = int(rng.integers(1, 4))
    p1 = (int(np.clip(x - dx, 1, w - 2)), int(np.clip(y - dy, 1, h - 2)))
    p2 = (int(np.clip(x + dx, 1, w - 2)), int(np.clip(y + dy, 1, h - 2)))
    cv2.line(mask, p1, p2, 255, width, cv2.LINE_AA)
    img[mask > 0] -= float(rng.uniform(35, 75))


def _chip(mask: np.ndarray, rng: np.random.Generator, corner: bool) -> None:
    """Render an edge/corner chip mask in canonical part coordinates."""
    h, w = mask.shape
    if corner:
        cx = int(rng.choice([5, w - 6]))
        cy = int(rng.choice([5, h - 6]))
    else:
        side = int(rng.integers(0, 4))
        cx = int(rng.integers(w // 4, 3 * w // 4))
        cy = int(rng.integers(h // 4, 3 * h // 4))
        if side == 0:
            cy = 3
        elif side == 1:
            cy = h - 4
        elif side == 2:
            cx = 3
        else:
            cx = w - 4
    cv2.circle(mask, (cx, cy), int(rng.integers(4, 10)), 255, -1)


def generate_sample(index: int, cfg: dict[str, Any], seed: int) -> Sample:
    """Generate one deterministic synthetic inspection image.

    Args:
        index: Dataset index.
        cfg: Project configuration.
        seed: Independent sample seed.
    Returns:
        Image, mask, class, bounding box and nuisance metadata.
    Raises:
        ValueError: Unsupported class.
    """
    d = cfg["dataset"]
    h, w = d["image_height_px"], d["image_width_px"]
    rng = np.random.default_rng(seed)
    bg = 35.0
    ph, pw = int(h * .58), int(w * .70)
    part = np.full((ph, pw), float(rng.uniform(150, 205)), np.float32)
    part += d["texture_strength"] * _texture(ph, pw, rng)
    _, xx = np.mgrid[0:ph, 0:pw]
    part += rng.uniform(-d["illumination_gradient"], d["illumination_gradient"]) * (xx / pw - .5)
    cls = d["classes"][index % len(d["classes"])]
    mask = np.zeros((ph, pw), np.uint8)
    if cls == "scratch":
        _scratch(part, mask, rng)
    elif cls == "edge_damage":
        _chip(mask, rng, False)
        part[mask > 0] = bg
    elif cls == "corner_damage":
        _chip(mask, rng, True)
        part[mask > 0] = bg
    elif cls == "mixed":
        _scratch(part, mask, rng)
        chip = np.zeros_like(mask)
        _chip(chip, rng, bool(index % 2))
        part[chip > 0] = bg
        mask = np.maximum(mask, chip)
    elif cls != "normal":
        raise ValueError(f"Unsupported class: {cls}")
    if rng.random() < d["marking_probability"]:
        cv2.putText(part, f"A{index % 100:02d}", (pw // 3, ph // 2), cv2.FONT_HERSHEY_SIMPLEX, .6, 110, 1, cv2.LINE_AA)
    canvas = np.full((h, w), bg, np.float32)
    fullmask = np.zeros((h, w), np.uint8)
    x0 = (w - pw) // 2
    y0 = (h - ph) // 2
    canvas[y0:y0 + ph, x0:x0 + pw] = part
    fullmask[y0:y0 + ph, x0:x0 + pw] = mask
    angle = float(rng.uniform(-cfg["application"]["max_rotation_deg"], cfg["application"]["max_rotation_deg"]))
    tx = float(rng.uniform(-d["translation_px"], d["translation_px"]))
    ty = float(rng.uniform(-d["translation_px"], d["translation_px"]))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    M[:, 2] += (tx, ty)
    canvas = cv2.warpAffine(canvas, M, (w, h), borderValue=bg)
    fullmask = cv2.warpAffine(fullmask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    if d["blur_sigma"] > 0:
        canvas = cv2.GaussianBlur(canvas, (0, 0), d["blur_sigma"])
    image = np.clip(canvas + rng.normal(0, d["noise_sigma"], canvas.shape), 0, 255).astype(np.uint8)
    ys, xs = np.where(fullmask > 0)
    bbox = None if not len(xs) else (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return Sample(image, fullmask, cls, bbox,
                  {"index": index, "seed": seed, "rotation_deg": angle, "translation_px": [tx, ty]})


def save_sample(sample: Sample, image_path: Path, mask_path: Path) -> None:
    """Save lossless image and mask.

    Raises:
        OSError: Directory/image cannot be written.
    """
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), sample.image) or not cv2.imwrite(str(mask_path), sample.mask):
        raise OSError("Failed to write generated sample")
