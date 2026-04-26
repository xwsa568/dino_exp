#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether huggan/night2day imageA/imageB pairs preserve DINO relation structure."
    )
    parser.add_argument("--dataset", default="huggan/night2day", help="HF dataset name.")
    parser.add_argument("--split", default="train", help="HF dataset split.")
    parser.add_argument("--column-a", default="imageA", help="Source/night image column.")
    parser.add_argument("--column-b", default="imageB", help="Target/day image column.")
    parser.add_argument("--num-samples", type=int, default=128, help="Number of rows to sample.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--streaming", dest="streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--model-repo", default="facebookresearch/dinov2")
    parser.add_argument("--model-name", default="dinov2_vitb14")
    parser.add_argument("--image-size", type=int, default=224, help="Square resize before DINO.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--token-subsample", type=int, default=0, help="0 uses all DINO patch tokens.")
    parser.add_argument("--tau", type=float, default=0.1, help="Temperature for second-order relation KL.")
    parser.add_argument("--remove-diag", action="store_true", help="Ignore self-similarity diagonal.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--torch-hub-dir", default=None, help="Optional torch.hub cache dir.")
    parser.add_argument("--output-dir", default="outputs/night2day_dino_probe")
    parser.add_argument("--num-grid", type=int, default=12, help="How many samples to show in pair_grid.png.")
    parser.add_argument("--num-heatmaps", type=int, default=4, help="How many samples to show in relation_heatmaps.png.")
    parser.add_argument("--num-overlay-samples", type=int, default=2, help="How many pairs to show in patch relation overlay plots.")
    parser.add_argument(
        "--overlay-anchors",
        default="center,upper_left,upper_right,lower_left,lower_right",
        help="Comma-separated anchor patch names or row:col coordinates for patch_relation_overlays.png.",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.45, help="Heatmap opacity for image overlays.")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_rgb_pil(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and "bytes" in value:
        import io

        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)}")


def load_samples(args: argparse.Namespace) -> Tuple[List[Image.Image], List[Image.Image]]:
    if args.num_samples < 2:
        raise ValueError("--num-samples must be >= 2 because shuffled negatives need a different row.")

    if args.streaming:
        ds = load_dataset(args.dataset, split=args.split, streaming=True)
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
        rows = []
        iterator = iter(ds)
        for _ in tqdm(range(args.num_samples), desc="Loading HF rows"):
            rows.append(next(iterator))
    else:
        ds = load_dataset(args.dataset, split=args.split)
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(ds)), k=min(args.num_samples, len(ds)))
        rows = [ds[i] for i in tqdm(indices, desc="Loading HF rows")]

    images_a = [to_rgb_pil(row[args.column_a]) for row in rows]
    images_b = [to_rgb_pil(row[args.column_b]) for row in rows]
    return images_a, images_b


def pil_to_dino_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def derangement_indices(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(1, n))
    return (np.arange(n) + shift) % n


def infer_patch_grid(num_tokens: int) -> Tuple[int, int]:
    side = int(round(float(num_tokens) ** 0.5))
    if side * side != num_tokens:
        raise ValueError(f"Expected a square patch-token grid, got {num_tokens} tokens.")
    return side, side


def parse_anchor_spec(spec: str, grid_h: int, grid_w: int) -> List[Tuple[str, int, int]]:
    named = {
        "center": (grid_h // 2, grid_w // 2),
        "upper_left": (grid_h // 4, grid_w // 4),
        "upper_right": (grid_h // 4, (3 * grid_w) // 4),
        "lower_left": ((3 * grid_h) // 4, grid_w // 4),
        "lower_right": ((3 * grid_h) // 4, (3 * grid_w) // 4),
        "top_left": (0, 0),
        "top_right": (0, grid_w - 1),
        "bottom_left": (grid_h - 1, 0),
        "bottom_right": (grid_h - 1, grid_w - 1),
    }
    anchors: List[Tuple[str, int, int]] = []
    seen = set()
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if item in named:
            row, col = named[item]
            label = item
        else:
            parts = item.split(":")
            if len(parts) != 2:
                raise ValueError(f"Unknown anchor '{item}'. Use a name like center or row:col.")
            row, col = int(parts[0]), int(parts[1])
            label = f"{row}:{col}"
        row = max(0, min(grid_h - 1, int(row)))
        col = max(0, min(grid_w - 1, int(col)))
        key = (row, col)
        if key not in seen:
            anchors.append((label, row, col))
            seen.add(key)
    if not anchors:
        raise ValueError("--overlay-anchors resolved to an empty anchor list.")
    return anchors


def maybe_subsample_tokens(
    a: torch.Tensor,
    b: torch.Tensor,
    token_subsample: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if token_subsample <= 0 or token_subsample >= a.shape[1]:
        return a, b
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    idx = torch.randperm(a.shape[1], generator=gen)[:token_subsample]
    return a[:, idx, :], b[:, idx, :]


def self_similarity(tokens: torch.Tensor, remove_diag: bool) -> torch.Tensor:
    x = F.normalize(tokens.float(), dim=-1, eps=1e-6)
    gram = torch.bmm(x, x.transpose(1, 2))
    if remove_diag:
        n = gram.shape[-1]
        eye = torch.eye(n, dtype=gram.dtype, device=gram.device).unsqueeze(0)
        gram = gram * (1.0 - eye)
    return gram


def second_order_gram(gram: torch.Tensor) -> torch.Tensor:
    rel = F.normalize(gram.float(), dim=-1, eps=1e-6)
    return torch.bmm(rel, rel.transpose(1, 2))


def kl_rows(ref_logits: torch.Tensor, pred_logits: torch.Tensor, tau: float) -> torch.Tensor:
    tau = max(float(tau), 1e-8)
    log_ref = F.log_softmax(ref_logits / tau, dim=-1)
    log_pred = F.log_softmax(pred_logits / tau, dim=-1)
    p_ref = log_ref.exp()
    return (p_ref * (log_ref - log_pred)).sum(dim=-1).mean(dim=-1)


def score_tokens(
    tokens_a: torch.Tensor,
    tokens_b: torch.Tensor,
    token_subsample: int,
    seed: int,
    tau: float,
    remove_diag: bool,
) -> Dict[str, np.ndarray]:
    tokens_a, tokens_b = maybe_subsample_tokens(tokens_a, tokens_b, token_subsample, seed)

    norm_a = F.normalize(tokens_a.float(), dim=-1, eps=1e-6)
    norm_b = F.normalize(tokens_b.float(), dim=-1, eps=1e-6)
    token_cos = (norm_a * norm_b).sum(dim=-1).mean(dim=-1)

    gram_a = self_similarity(tokens_a, remove_diag=remove_diag)
    gram_b = self_similarity(tokens_b, remove_diag=remove_diag)
    rel_cos = F.cosine_similarity(gram_a.flatten(1), gram_b.flatten(1), dim=-1)
    rel_mse = F.mse_loss(gram_a, gram_b, reduction="none").flatten(1).mean(dim=-1)

    second_a = second_order_gram(gram_a)
    second_b = second_order_gram(gram_b)
    rel_2gram_skl = 0.5 * (kl_rows(second_a, second_b, tau=tau) + kl_rows(second_b, second_a, tau=tau))

    return {
        "token_cos": token_cos.cpu().numpy(),
        "dino_rel_cos": rel_cos.cpu().numpy(),
        "dino_rel_mse": rel_mse.cpu().numpy(),
        "dino_rel_2gram_skl": rel_2gram_skl.cpu().numpy(),
    }


class DinoExtractor:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        if args.torch_hub_dir:
            torch.hub.set_dir(args.torch_hub_dir)
        self.device = device
        self.image_size = args.image_size
        self.net = torch.hub.load(args.model_repo, args.model_name)
        self.net.eval().to(device)
        self.net.requires_grad_(False)

    @torch.no_grad()
    def extract(self, images: Sequence[Image.Image], batch_size: int) -> torch.Tensor:
        batches = []
        for start in tqdm(range(0, len(images), batch_size), desc="Extracting DINO tokens"):
            chunk = images[start : start + batch_size]
            x = torch.stack([pil_to_dino_tensor(img, self.image_size) for img in chunk]).to(self.device)
            out = self.net.forward_features(x)
            if not isinstance(out, dict) or "x_norm_patchtokens" not in out:
                raise RuntimeError("DINOv2 forward_features did not return x_norm_patchtokens.")
            batches.append(out["x_norm_patchtokens"].detach().cpu())
        return torch.cat(batches, dim=0)


def bootstrap_ci(
    paired: np.ndarray,
    shuffled: np.ndarray,
    higher_is_better: bool,
    reps: int,
    seed: int,
) -> Optional[Tuple[float, float]]:
    if reps <= 0:
        return None
    rng = np.random.default_rng(seed)
    n = len(paired)
    improvements = []
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        if higher_is_better:
            improvements.append(float(np.mean(paired[idx] - shuffled[idx])))
        else:
            improvements.append(float(np.mean(shuffled[idx] - paired[idx])))
    return float(np.percentile(improvements, 2.5)), float(np.percentile(improvements, 97.5))


def summarize_metric(
    paired: np.ndarray,
    shuffled: np.ndarray,
    higher_is_better: bool,
    reps: int,
    seed: int,
) -> Dict[str, float]:
    if higher_is_better:
        improvement = paired - shuffled
        better = paired > shuffled
    else:
        improvement = shuffled - paired
        better = paired < shuffled
    ci = bootstrap_ci(paired, shuffled, higher_is_better, reps, seed)
    summary = {
        "paired_mean": float(np.mean(paired)),
        "paired_std": float(np.std(paired, ddof=1)),
        "shuffled_mean": float(np.mean(shuffled)),
        "shuffled_std": float(np.std(shuffled, ddof=1)),
        "improvement_mean": float(np.mean(improvement)),
        "paired_better_rate": float(np.mean(better)),
    }
    if ci is not None:
        summary["improvement_ci95_low"] = ci[0]
        summary["improvement_ci95_high"] = ci[1]
    return summary


def save_scores_csv(
    path: Path,
    paired_scores: Dict[str, np.ndarray],
    shuffled_scores: Dict[str, np.ndarray],
    shuffled_indices: np.ndarray,
) -> None:
    metrics = list(paired_scores.keys())
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["idx", "shuffled_b_idx"]
        for metric in metrics:
            header.extend([f"paired_{metric}", f"shuffled_{metric}"])
        writer.writerow(header)
        for i in range(len(shuffled_indices)):
            row = [i, int(shuffled_indices[i])]
            for metric in metrics:
                row.extend([float(paired_scores[metric][i]), float(shuffled_scores[metric][i])])
            writer.writerow(row)


def save_distribution_plot(
    path: Path,
    paired_scores: Dict[str, np.ndarray],
    shuffled_scores: Dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    specs = [
        ("dino_rel_cos", "DINO relation cosine (higher is better)"),
        ("dino_rel_mse", "DINO relation MSE (lower is better)"),
        ("dino_rel_2gram_skl", "Second-order relation KL (lower is better)"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        ax.hist(paired_scores[metric], bins=24, alpha=0.65, label="paired")
        ax.hist(shuffled_scores[metric], bins=24, alpha=0.65, label="shuffled")
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.set_ylabel("count")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str) -> None:
    try:
        font = ImageFont.truetype("Arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=(20, 20, 20), font=font)


def save_pair_grid(
    path: Path,
    images_a: Sequence[Image.Image],
    images_b: Sequence[Image.Image],
    shuffled_indices: np.ndarray,
    paired_scores: Dict[str, np.ndarray],
    shuffled_scores: Dict[str, np.ndarray],
    num_grid: int,
) -> None:
    n = min(num_grid, len(images_a))
    tile = 160
    label_h = 54
    cols = 3
    rows = n
    canvas = Image.new("RGB", (cols * tile, rows * (tile + label_h)), "white")
    draw = ImageDraw.Draw(canvas)

    for r in range(n):
        triplet = [images_a[r], images_b[r], images_b[int(shuffled_indices[r])]]
        labels = [
            f"A idx={r}",
            f"true B rel={paired_scores['dino_rel_cos'][r]:.3f}",
            f"shuf B rel={shuffled_scores['dino_rel_cos'][r]:.3f}",
        ]
        for c, (img, label) in enumerate(zip(triplet, labels)):
            x = c * tile
            y = r * (tile + label_h)
            thumb = img.convert("RGB").resize((tile, tile), Image.Resampling.BICUBIC)
            canvas.paste(thumb, (x, y))
            draw_label(draw, (x + 4, y + tile + 4), label)
            if c == 1:
                draw_label(draw, (x + 4, y + tile + 22), f"2gram={paired_scores['dino_rel_2gram_skl'][r]:.4f}")
            if c == 2:
                draw_label(draw, (x + 4, y + tile + 22), f"2gram={shuffled_scores['dino_rel_2gram_skl'][r]:.4f}")

    canvas.save(path)


def save_relation_heatmaps(
    path: Path,
    tokens_a: torch.Tensor,
    tokens_b: torch.Tensor,
    shuffled_indices: np.ndarray,
    num_heatmaps: int,
    remove_diag: bool,
) -> None:
    n = min(num_heatmaps, tokens_a.shape[0])
    fig, axes = plt.subplots(n, 5, figsize=(16, 3.2 * n), squeeze=False)
    for r in range(n):
        gram_a = self_similarity(tokens_a[r : r + 1], remove_diag=remove_diag)[0].numpy()
        gram_b = self_similarity(tokens_b[r : r + 1], remove_diag=remove_diag)[0].numpy()
        gram_s = self_similarity(tokens_b[int(shuffled_indices[r]) : int(shuffled_indices[r]) + 1], remove_diag=remove_diag)[0].numpy()
        panels = [
            (gram_a, f"A {r}"),
            (gram_b, "true B"),
            (gram_s, f"shuf B {int(shuffled_indices[r])}"),
            (np.abs(gram_a - gram_b), "|A - true B|"),
            (np.abs(gram_a - gram_s), "|A - shuf B|"),
        ]
        for c, (mat, title) in enumerate(panels):
            ax = axes[r][c]
            cmap = "viridis" if c < 3 else "magma"
            ax.imshow(mat, cmap=cmap, vmin=None, vmax=None)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_image_array(image: Image.Image, image_size: int) -> np.ndarray:
    image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def add_anchor_box(ax, row: int, col: int, grid_h: int, grid_w: int, image_size: int) -> None:
    patch_h = float(image_size) / float(grid_h)
    patch_w = float(image_size) / float(grid_w)
    rect = plt.Rectangle(
        (col * patch_w, row * patch_h),
        patch_w,
        patch_h,
        fill=False,
        edgecolor="cyan",
        linewidth=1.5,
    )
    ax.add_patch(rect)


def vector_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def save_patch_relation_overlays(
    path: Path,
    images_a: Sequence[Image.Image],
    images_b: Sequence[Image.Image],
    tokens_a: torch.Tensor,
    tokens_b: torch.Tensor,
    num_samples: int,
    anchor_spec: str,
    image_size: int,
    alpha: float,
    remove_diag: bool,
) -> None:
    n = min(max(0, num_samples), len(images_a))
    if n == 0:
        return

    grid_h, grid_w = infer_patch_grid(tokens_a.shape[1])
    anchors = parse_anchor_spec(anchor_spec, grid_h, grid_w)
    rows = n * len(anchors)
    fig, axes = plt.subplots(rows, 5, figsize=(17, max(3.0, 2.7 * rows)), squeeze=False)

    for sample_idx in range(n):
        img_a = plot_image_array(images_a[sample_idx], image_size)
        img_b = plot_image_array(images_b[sample_idx], image_size)
        gram_a = self_similarity(tokens_a[sample_idx : sample_idx + 1], remove_diag=remove_diag)[0].numpy()
        gram_b = self_similarity(tokens_b[sample_idx : sample_idx + 1], remove_diag=remove_diag)[0].numpy()

        for anchor_offset, (anchor_label, patch_row, patch_col) in enumerate(anchors):
            row_idx = sample_idx * len(anchors) + anchor_offset
            token_idx = patch_row * grid_w + patch_col
            rel_a = gram_a[token_idx].reshape(grid_h, grid_w)
            rel_b = gram_b[token_idx].reshape(grid_h, grid_w)
            rel_diff = np.abs(rel_a - rel_b)
            row_cos = vector_cosine(rel_a, rel_b)
            row_mse = float(np.mean((rel_a - rel_b) ** 2))

            panels = [
                ("A image", img_a, None, None, None),
                ("B image", img_b, None, None, None),
                ("A anchor relation", img_a, rel_a, "magma", (-1.0, 1.0)),
                ("B anchor relation", img_b, rel_b, "magma", (-1.0, 1.0)),
                ("abs relation diff", img_a, rel_diff, "viridis", (0.0, 1.0)),
            ]
            for col_idx, (title, base, heat, cmap, limits) in enumerate(panels):
                ax = axes[row_idx][col_idx]
                ax.imshow(base)
                if heat is not None:
                    vmin, vmax = limits
                    ax.imshow(
                        heat,
                        cmap=cmap,
                        alpha=alpha,
                        interpolation="bilinear",
                        extent=(0, image_size, image_size, 0),
                        vmin=vmin,
                        vmax=vmax,
                    )
                add_anchor_box(ax, patch_row, patch_col, grid_h, grid_w, image_size)
                if row_idx == 0:
                    ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])

            axes[row_idx][0].set_ylabel(
                f"sample {sample_idx}\n{anchor_label} ({patch_row},{patch_col})",
                rotation=0,
                labelpad=48,
                va="center",
            )
            axes[row_idx][4].set_xlabel(f"row cos={row_cos:.3f}, row mse={row_mse:.4f}")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_patch_relation_similarity_maps(
    path: Path,
    images_a: Sequence[Image.Image],
    images_b: Sequence[Image.Image],
    tokens_a: torch.Tensor,
    tokens_b: torch.Tensor,
    num_samples: int,
    image_size: int,
    alpha: float,
    remove_diag: bool,
) -> None:
    n = min(max(0, num_samples), len(images_a))
    if n == 0:
        return

    grid_h, grid_w = infer_patch_grid(tokens_a.shape[1])
    fig, axes = plt.subplots(n, 4, figsize=(14, max(3.0, 3.2 * n)), squeeze=False)

    for sample_idx in range(n):
        img_a = plot_image_array(images_a[sample_idx], image_size)
        img_b = plot_image_array(images_b[sample_idx], image_size)
        gram_a = self_similarity(tokens_a[sample_idx : sample_idx + 1], remove_diag=remove_diag)[0]
        gram_b = self_similarity(tokens_b[sample_idx : sample_idx + 1], remove_diag=remove_diag)[0]
        row_cos = F.cosine_similarity(gram_a.float(), gram_b.float(), dim=-1, eps=1e-6).numpy().reshape(grid_h, grid_w)
        row_mse = ((gram_a - gram_b) ** 2).mean(dim=-1).numpy().reshape(grid_h, grid_w)
        mse_vmax = max(float(np.percentile(row_mse, 95)), 1e-6)

        panels = [
            ("A image", img_a, None, None, None),
            ("B image", img_b, None, None, None),
            (f"per-anchor row cos mean={row_cos.mean():.3f}", img_a, row_cos, "magma", (0.0, 1.0)),
            (f"per-anchor row mse mean={row_mse.mean():.4f}", img_a, row_mse, "viridis", (0.0, mse_vmax)),
        ]
        for col_idx, (title, base, heat, cmap, limits) in enumerate(panels):
            ax = axes[sample_idx][col_idx]
            ax.imshow(base)
            if heat is not None:
                vmin, vmax = limits
                ax.imshow(
                    heat,
                    cmap=cmap,
                    alpha=alpha,
                    interpolation="bilinear",
                    extent=(0, image_size, image_size, 0),
                    vmin=vmin,
                    vmax=vmax,
                )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[sample_idx][0].set_ylabel(f"sample {sample_idx}", rotation=0, labelpad=35, va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"device={device}")
    print(f"dataset={args.dataset} split={args.split} columns=({args.column_a}, {args.column_b})")

    images_a, images_b = load_samples(args)
    shuffled_indices = derangement_indices(len(images_a), args.seed + 17)

    extractor = DinoExtractor(args, device)
    tokens_a = extractor.extract(images_a, args.batch_size)
    tokens_b = extractor.extract(images_b, args.batch_size)

    paired_scores = score_tokens(
        tokens_a,
        tokens_b,
        token_subsample=args.token_subsample,
        seed=args.seed,
        tau=args.tau,
        remove_diag=args.remove_diag,
    )
    shuffled_scores = score_tokens(
        tokens_a,
        tokens_b[torch.from_numpy(shuffled_indices).long()],
        token_subsample=args.token_subsample,
        seed=args.seed,
        tau=args.tau,
        remove_diag=args.remove_diag,
    )

    higher_is_better = {
        "token_cos": True,
        "dino_rel_cos": True,
        "dino_rel_mse": False,
        "dino_rel_2gram_skl": False,
    }
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "column_a": args.column_a,
        "column_b": args.column_b,
        "num_samples": len(images_a),
        "model_repo": args.model_repo,
        "model_name": args.model_name,
        "image_size": args.image_size,
        "patch_grid": list(infer_patch_grid(tokens_a.shape[1])),
        "token_subsample": args.token_subsample,
        "remove_diag": args.remove_diag,
        "overlay_anchors": args.overlay_anchors,
        "shuffled_indices_rule": "cyclic random shift derangement",
        "metrics": {},
    }
    for metric, hib in higher_is_better.items():
        summary["metrics"][metric] = summarize_metric(
            paired_scores[metric],
            shuffled_scores[metric],
            higher_is_better=hib,
            reps=args.bootstrap_reps,
            seed=args.seed + 101,
        )

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    save_scores_csv(output_dir / "scores.csv", paired_scores, shuffled_scores, shuffled_indices)
    save_distribution_plot(output_dir / "score_distributions.png", paired_scores, shuffled_scores)
    save_pair_grid(output_dir / "pair_grid.png", images_a, images_b, shuffled_indices, paired_scores, shuffled_scores, args.num_grid)
    if args.num_heatmaps > 0:
        save_relation_heatmaps(output_dir / "relation_heatmaps.png", tokens_a, tokens_b, shuffled_indices, args.num_heatmaps, args.remove_diag)
    save_patch_relation_overlays(
        output_dir / "patch_relation_overlays.png",
        images_a,
        images_b,
        tokens_a,
        tokens_b,
        args.num_overlay_samples,
        args.overlay_anchors,
        args.image_size,
        args.overlay_alpha,
        args.remove_diag,
    )
    save_patch_relation_similarity_maps(
        output_dir / "patch_relation_similarity_maps.png",
        images_a,
        images_b,
        tokens_a,
        tokens_b,
        args.num_overlay_samples,
        args.image_size,
        args.overlay_alpha,
        args.remove_diag,
    )

    print(json.dumps(summary["metrics"], indent=2))
    print(f"saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
