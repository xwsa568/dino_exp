#!/usr/bin/env python3
import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import get_dataset_split_names, load_dataset
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether huggan/night2day imageA/imageB pairs preserve DINO relation structure."
    )
    parser.add_argument("--dataset", default="huggan/night2day", help="HF dataset name.")
    parser.add_argument(
        "--split",
        default="auto",
        help="HF dataset split. 'auto' uses test if present, then validation, then train.",
    )
    parser.add_argument("--column-a", default="imageA", help="Source/night image column.")
    parser.add_argument("--column-b", default="imageB", help="Target/day image column.")
    parser.add_argument("--num-samples", type=int, default=0, help="Number of rows to sample. 0 uses the full split.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--streaming", dest="streaming", action="store_true", default=False)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--model-repo", default="facebookresearch/dinov2")
    parser.add_argument("--model-name", default="dinov2_vitb14")
    parser.add_argument("--image-size", type=int, default=224, help="Square resize before DINO.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--token-subsample", type=int, default=0, help="Random token subsample for paired-vs-shuffled metrics. 0 uses all extracted tokens.")
    parser.add_argument(
        "--metric-token-grid",
        type=int,
        default=8,
        help="Use an evenly spaced k x k DINO token grid for scalable full retrieval. 0 uses all patch tokens.",
    )
    parser.add_argument("--tau", type=float, default=0.1, help="Temperature for second-order relation KL.")
    parser.add_argument("--remove-diag", action="store_true", help="Ignore self-similarity diagonal.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument(
        "--ranking-device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device for all-B retrieval scoring. Defaults to the extraction device.",
    )
    parser.add_argument("--ranking-query-batch-size", type=int, default=8)
    parser.add_argument("--ranking-candidate-batch-size", type=int, default=128)
    parser.add_argument("--pixel-size", type=int, default=64, help="Square resize for pixel-level retrieval metrics.")
    parser.add_argument("--storage-dtype", default="float16", choices=["float16", "float32"], help="CPU storage dtype for cached representations.")
    parser.add_argument(
        "--retrieval-metrics",
        default="all",
        help="Comma-separated retrieval metrics or 'all'. Names are written to retrieval_summary.json.",
    )
    parser.add_argument("--skip-retrieval", action="store_true", help="Only run paired-vs-shuffled summaries and plots.")
    parser.add_argument("--skip-paired-shuffled", action="store_true", help="Only run full all-B retrieval ranking.")
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


def resolve_split(dataset: str, split: str) -> str:
    if split != "auto":
        return split
    split_names = get_dataset_split_names(dataset)
    for preferred in ("test", "validation", "val", "train"):
        if preferred in split_names:
            return preferred
    if not split_names:
        raise ValueError(f"No splits found for dataset '{dataset}'.")
    return split_names[0]


def storage_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported storage dtype: {name}")


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
    if args.num_samples < 0:
        raise ValueError("--num-samples must be >= 0. Use 0 for the full split.")

    if args.streaming:
        ds = load_dataset(args.dataset, split=args.split, streaming=True)
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
        rows = []
        iterator = iter(ds)
        if args.num_samples == 0:
            for row in tqdm(iterator, desc="Loading HF rows"):
                rows.append(row)
        else:
            for _ in tqdm(range(args.num_samples), desc="Loading HF rows"):
                rows.append(next(iterator))
    else:
        ds = load_dataset(args.dataset, split=args.split)
        rng = random.Random(args.seed)
        if args.num_samples == 0:
            indices = list(range(len(ds)))
        else:
            indices = rng.sample(range(len(ds)), k=min(args.num_samples, len(ds)))
        rows = [ds[i] for i in tqdm(indices, desc="Loading HF rows")]

    if len(rows) < 2:
        raise ValueError("Need at least two rows because shuffled negatives and retrieval ranks need candidates.")

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


def spatial_token_indices(num_tokens: int, metric_token_grid: int) -> Tuple[Optional[torch.Tensor], Tuple[int, int]]:
    grid_h, grid_w = infer_patch_grid(num_tokens)
    if metric_token_grid <= 0 or metric_token_grid >= min(grid_h, grid_w):
        return None, (grid_h, grid_w)
    rows = np.floor((np.arange(metric_token_grid) + 0.5) * grid_h / metric_token_grid).astype(int)
    cols = np.floor((np.arange(metric_token_grid) + 0.5) * grid_w / metric_token_grid).astype(int)
    rows = np.clip(rows, 0, grid_h - 1)
    cols = np.clip(cols, 0, grid_w - 1)
    idx = [int(r * grid_w + c) for r in rows for c in cols]
    return torch.tensor(idx, dtype=torch.long), (metric_token_grid, metric_token_grid)


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


def symmetric_kl_rows(a_logits: torch.Tensor, b_logits: torch.Tensor, tau: float) -> torch.Tensor:
    return 0.5 * (kl_rows(a_logits, b_logits, tau=tau) + kl_rows(b_logits, a_logits, tau=tau))


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
    token_l1 = (tokens_a.float() - tokens_b.float()).abs().flatten(1).mean(dim=-1)
    token_l2 = F.mse_loss(tokens_a.float(), tokens_b.float(), reduction="none").flatten(1).mean(dim=-1)

    gram_a = self_similarity(tokens_a, remove_diag=remove_diag)
    gram_b = self_similarity(tokens_b, remove_diag=remove_diag)
    rel1_cos = F.cosine_similarity(gram_a.flatten(1), gram_b.flatten(1), dim=-1)
    rel1_l1 = (gram_a - gram_b).abs().flatten(1).mean(dim=-1)
    rel1_l2 = F.mse_loss(gram_a, gram_b, reduction="none").flatten(1).mean(dim=-1)
    rel1_skl = symmetric_kl_rows(gram_a, gram_b, tau=tau)

    second_a = second_order_gram(gram_a)
    second_b = second_order_gram(gram_b)
    rel2_cos = F.cosine_similarity(second_a.flatten(1), second_b.flatten(1), dim=-1)
    rel2_l1 = (second_a - second_b).abs().flatten(1).mean(dim=-1)
    rel2_l2 = F.mse_loss(second_a, second_b, reduction="none").flatten(1).mean(dim=-1)
    rel2_skl = symmetric_kl_rows(second_a, second_b, tau=tau)
    return {
        "token_cos": token_cos.cpu().numpy(),
        "dino_token_l1": token_l1.cpu().numpy(),
        "dino_token_l2": token_l2.cpu().numpy(),
        "dino_rel1_cos": rel1_cos.cpu().numpy(),
        "dino_rel1_l1": rel1_l1.cpu().numpy(),
        "dino_rel1_l2": rel1_l2.cpu().numpy(),
        "dino_rel1_skl": rel1_skl.cpu().numpy(),
        "dino_rel2_cos": rel2_cos.cpu().numpy(),
        "dino_rel2_l1": rel2_l1.cpu().numpy(),
        "dino_rel2_l2": rel2_l2.cpu().numpy(),
        "dino_rel2_skl": rel2_skl.cpu().numpy(),
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
    def extract(
        self,
        images: Sequence[Image.Image],
        batch_size: int,
        metric_token_grid: int,
        out_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        batches = []
        token_indices = None
        selected_grid = None
        for start in tqdm(range(0, len(images), batch_size), desc="Extracting DINO tokens"):
            chunk = images[start : start + batch_size]
            x = torch.stack([pil_to_dino_tensor(img, self.image_size) for img in chunk]).to(self.device)
            out = self.net.forward_features(x)
            if not isinstance(out, dict) or "x_norm_patchtokens" not in out:
                raise RuntimeError("DINOv2 forward_features did not return x_norm_patchtokens.")
            tokens = out["x_norm_patchtokens"]
            if selected_grid is None:
                token_indices, selected_grid = spatial_token_indices(tokens.shape[1], metric_token_grid)
                if token_indices is not None:
                    token_indices = token_indices.to(tokens.device)
            if token_indices is not None:
                tokens = tokens[:, token_indices, :]
            batches.append(tokens.detach().cpu().to(out_dtype))
        if selected_grid is None:
            raise RuntimeError("No images were passed to DINO extractor.")
        return torch.cat(batches, dim=0), selected_grid


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


ALL_RETRIEVAL_METRICS = [
    "pixel_l1",
    "pixel_l2",
    "dino_token_cos",
    "dino_token_l1",
    "dino_token_l2",
    "dino_rel1_cos",
    "dino_rel1_l1",
    "dino_rel1_l2",
    "dino_rel1_skl",
    "dino_rel2_cos",
    "dino_rel2_l1",
    "dino_rel2_l2",
    "dino_rel2_skl",
]


@dataclass
class RetrievalResult:
    ranks: np.ndarray
    true_scores: np.ndarray
    higher_is_better: bool


@dataclass
class KlCache:
    probs: torch.Tensor
    log_probs: torch.Tensor
    const: torch.Tensor
    row_count: int


def parse_retrieval_metrics(spec: str) -> List[str]:
    if spec.strip().lower() == "all":
        return list(ALL_RETRIEVAL_METRICS)
    metrics = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = sorted(set(metrics) - set(ALL_RETRIEVAL_METRICS))
    if unknown:
        raise ValueError(f"Unknown retrieval metrics: {unknown}. Valid metrics: {ALL_RETRIEVAL_METRICS}")
    if not metrics:
        raise ValueError("--retrieval-metrics resolved to an empty metric list.")
    return metrics


def images_to_pixel_vectors(images: Sequence[Image.Image], pixel_size: int, dtype: torch.dtype) -> torch.Tensor:
    vectors = []
    for image in tqdm(images, desc="Building pixel vectors"):
        image = image.convert("RGB").resize((pixel_size, pixel_size), Image.Resampling.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        vectors.append(torch.from_numpy(arr).flatten())
    return torch.stack(vectors).to(dtype)


def flatten_batches(x: torch.Tensor, dtype: torch.dtype, batch_size: int, desc: str) -> torch.Tensor:
    batches = []
    for start in tqdm(range(0, x.shape[0], batch_size), desc=desc):
        batches.append(x[start : start + batch_size].float().flatten(1).cpu().to(dtype))
    return torch.cat(batches, dim=0)


def normalized_flat_batches(x: torch.Tensor, dtype: torch.dtype, batch_size: int, desc: str) -> torch.Tensor:
    batches = []
    for start in tqdm(range(0, x.shape[0], batch_size), desc=desc):
        flat = x[start : start + batch_size].float().flatten(1)
        batches.append(F.normalize(flat, dim=-1, eps=1e-6).cpu().to(dtype))
    return torch.cat(batches, dim=0)


def normalized_token_flat_batches(x: torch.Tensor, dtype: torch.dtype, batch_size: int, desc: str) -> torch.Tensor:
    batches = []
    for start in tqdm(range(0, x.shape[0], batch_size), desc=desc):
        tokens = x[start : start + batch_size].float()
        batches.append(F.normalize(tokens, dim=-1, eps=1e-6).flatten(1).cpu().to(dtype))
    return torch.cat(batches, dim=0)


def build_relation_flats(
    tokens: torch.Tensor,
    remove_diag: bool,
    dtype: torch.dtype,
    batch_size: int,
    build_second: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    rel1_batches = []
    rel2_batches = []
    for start in tqdm(range(0, tokens.shape[0], batch_size), desc="Building relation matrices"):
        gram = self_similarity(tokens[start : start + batch_size], remove_diag=remove_diag)
        rel1_batches.append(gram.flatten(1).cpu().to(dtype))
        if build_second:
            rel2_batches.append(second_order_gram(gram).flatten(1).cpu().to(dtype))
    rel1 = torch.cat(rel1_batches, dim=0)
    rel2 = torch.cat(rel2_batches, dim=0) if build_second else None
    return rel1, rel2


def make_kl_cache(matrix_flat: torch.Tensor, row_count: int, tau: float, dtype: torch.dtype, batch_size: int) -> KlCache:
    prob_batches = []
    log_prob_batches = []
    const_batches = []
    tau = max(float(tau), 1e-8)
    for start in tqdm(range(0, matrix_flat.shape[0], batch_size), desc="Building KL caches"):
        logits = matrix_flat[start : start + batch_size].float().view(-1, row_count, row_count) / tau
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        const = (probs * log_probs).sum(dim=-1).mean(dim=-1)
        prob_batches.append(probs.flatten(1).cpu().to(dtype))
        log_prob_batches.append(log_probs.flatten(1).cpu().to(dtype))
        const_batches.append(const.cpu())
    return KlCache(
        probs=torch.cat(prob_batches, dim=0),
        log_probs=torch.cat(log_prob_batches, dim=0),
        const=torch.cat(const_batches, dim=0),
        row_count=row_count,
    )


def aligned_vector_scores(
    a: torch.Tensor,
    b: torch.Tensor,
    kind: str,
    device: torch.device,
    batch_size: int,
    scale: float = 1.0,
) -> np.ndarray:
    scores = []
    dim = a.shape[1]
    for start in range(0, a.shape[0], batch_size):
        qa = a[start : start + batch_size].to(device=device, dtype=torch.float32)
        qb = b[start : start + batch_size].to(device=device, dtype=torch.float32)
        if kind == "cos":
            score = (qa * qb).sum(dim=-1) * scale
        elif kind == "l1":
            score = (qa - qb).abs().mean(dim=-1)
        elif kind == "l2":
            score = ((qa - qb) ** 2).mean(dim=-1)
        else:
            raise ValueError(f"Unsupported vector score kind: {kind}")
        scores.append(score.detach().cpu())
    return torch.cat(scores).numpy()


def vector_score_block(q: torch.Tensor, c: torch.Tensor, kind: str, scale: float = 1.0) -> torch.Tensor:
    if kind == "cos":
        return torch.mm(q, c.transpose(0, 1)) * scale
    if kind == "l1":
        return (q[:, None, :] - c[None, :, :]).abs().mean(dim=-1)
    if kind == "l2":
        dim = q.shape[1]
        q_norm = (q * q).sum(dim=-1, keepdim=True)
        c_norm = (c * c).sum(dim=-1).unsqueeze(0)
        return (q_norm + c_norm - 2.0 * torch.mm(q, c.transpose(0, 1))).clamp_min(0.0) / float(dim)
    raise ValueError(f"Unsupported vector score kind: {kind}")


@torch.no_grad()
def rank_vector_metric(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    kind: str,
    higher_is_better: bool,
    device: torch.device,
    query_batch_size: int,
    candidate_batch_size: int,
    scale: float = 1.0,
) -> RetrievalResult:
    if a.shape != b.shape:
        raise ValueError(f"{name}: A/B representation shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}")
    n = a.shape[0]
    true_scores = aligned_vector_scores(a, b, kind, device, max(1, query_batch_size), scale=scale)
    better_counts = np.zeros(n, dtype=np.int64)
    for q_start in tqdm(range(0, n, query_batch_size), desc=f"Ranking {name}"):
        q_end = min(q_start + query_batch_size, n)
        q = a[q_start:q_end].to(device=device, dtype=torch.float32)
        true = torch.from_numpy(true_scores[q_start:q_end]).to(device=device, dtype=torch.float32)
        counts = torch.zeros(q_end - q_start, dtype=torch.int64)
        for c_start in range(0, n, candidate_batch_size):
            c_end = min(c_start + candidate_batch_size, n)
            c = b[c_start:c_end].to(device=device, dtype=torch.float32)
            scores = vector_score_block(q, c, kind, scale=scale)
            overlap_start = max(q_start, c_start)
            overlap_end = min(q_end, c_end)
            if overlap_start < overlap_end:
                q_idx = torch.arange(overlap_start - q_start, overlap_end - q_start, device=device)
                c_idx = torch.arange(overlap_start - c_start, overlap_end - c_start, device=device)
                scores[q_idx, c_idx] = true[q_idx]
            if higher_is_better:
                counts += (scores > true[:, None]).sum(dim=1).detach().cpu()
            else:
                counts += (scores < true[:, None]).sum(dim=1).detach().cpu()
        better_counts[q_start:q_end] = counts.numpy()
    return RetrievalResult(ranks=better_counts + 1, true_scores=true_scores, higher_is_better=higher_is_better)


def aligned_skl_scores(
    a: KlCache,
    b: KlCache,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    scores = []
    rows = float(a.row_count)
    for start in range(0, a.probs.shape[0], batch_size):
        p_a = a.probs[start : start + batch_size].to(device=device, dtype=torch.float32)
        lp_a = a.log_probs[start : start + batch_size].to(device=device, dtype=torch.float32)
        c_a = a.const[start : start + batch_size].to(device=device, dtype=torch.float32)
        p_b = b.probs[start : start + batch_size].to(device=device, dtype=torch.float32)
        lp_b = b.log_probs[start : start + batch_size].to(device=device, dtype=torch.float32)
        c_b = b.const[start : start + batch_size].to(device=device, dtype=torch.float32)
        kl_ab = c_a - (p_a * lp_b).sum(dim=-1) / rows
        kl_ba = c_b - (p_b * lp_a).sum(dim=-1) / rows
        scores.append((0.5 * (kl_ab + kl_ba)).detach().cpu())
    return torch.cat(scores).numpy()


def skl_score_block(a: KlCache, b: KlCache, q_slice: slice, c_slice: slice, device: torch.device) -> torch.Tensor:
    rows = float(a.row_count)
    p_a = a.probs[q_slice].to(device=device, dtype=torch.float32)
    lp_a = a.log_probs[q_slice].to(device=device, dtype=torch.float32)
    c_a = a.const[q_slice].to(device=device, dtype=torch.float32)
    p_b = b.probs[c_slice].to(device=device, dtype=torch.float32)
    lp_b = b.log_probs[c_slice].to(device=device, dtype=torch.float32)
    c_b = b.const[c_slice].to(device=device, dtype=torch.float32)
    kl_ab = c_a[:, None] - torch.mm(p_a, lp_b.transpose(0, 1)) / rows
    kl_ba = c_b[None, :] - torch.mm(p_b, lp_a.transpose(0, 1)).transpose(0, 1) / rows
    return 0.5 * (kl_ab + kl_ba)


@torch.no_grad()
def rank_skl_metric(
    name: str,
    a: KlCache,
    b: KlCache,
    device: torch.device,
    query_batch_size: int,
    candidate_batch_size: int,
) -> RetrievalResult:
    if a.probs.shape != b.probs.shape:
        raise ValueError(f"{name}: A/B KL cache shapes differ: {tuple(a.probs.shape)} vs {tuple(b.probs.shape)}")
    n = a.probs.shape[0]
    true_scores = aligned_skl_scores(a, b, device, max(1, query_batch_size))
    better_counts = np.zeros(n, dtype=np.int64)
    for q_start in tqdm(range(0, n, query_batch_size), desc=f"Ranking {name}"):
        q_end = min(q_start + query_batch_size, n)
        true = torch.from_numpy(true_scores[q_start:q_end]).to(device=device, dtype=torch.float32)
        counts = torch.zeros(q_end - q_start, dtype=torch.int64)
        q_slice = slice(q_start, q_end)
        for c_start in range(0, n, candidate_batch_size):
            c_end = min(c_start + candidate_batch_size, n)
            scores = skl_score_block(a, b, q_slice, slice(c_start, c_end), device)
            overlap_start = max(q_start, c_start)
            overlap_end = min(q_end, c_end)
            if overlap_start < overlap_end:
                q_idx = torch.arange(overlap_start - q_start, overlap_end - q_start, device=device)
                c_idx = torch.arange(overlap_start - c_start, overlap_end - c_start, device=device)
                scores[q_idx, c_idx] = true[q_idx]
            counts += (scores < true[:, None]).sum(dim=1).detach().cpu()
        better_counts[q_start:q_end] = counts.numpy()
    return RetrievalResult(ranks=better_counts + 1, true_scores=true_scores, higher_is_better=False)


def summarize_retrieval_result(result: RetrievalResult, candidate_count: int) -> Dict[str, float]:
    ranks = result.ranks.astype(np.float64)
    denom = max(candidate_count - 1, 1)
    return {
        "candidate_count": int(candidate_count),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "top1_rate": float(np.mean(ranks <= 1)),
        "top5_rate": float(np.mean(ranks <= min(5, candidate_count))),
        "top10_rate": float(np.mean(ranks <= min(10, candidate_count))),
        "top1pct_rate": float(np.mean(ranks <= max(1, int(np.ceil(candidate_count * 0.01))))),
        "mean_pair_percentile": float(np.mean(1.0 - (ranks - 1.0) / denom)),
        "true_score_mean": float(np.mean(result.true_scores)),
        "true_score_std": float(np.std(result.true_scores, ddof=1)),
        "higher_is_better": bool(result.higher_is_better),
    }


def save_retrieval_ranks_csv(path: Path, results: Dict[str, RetrievalResult]) -> None:
    metrics = list(results.keys())
    n = len(next(iter(results.values())).ranks)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["idx"]
        for metric in metrics:
            header.extend([f"{metric}_rank", f"{metric}_true_score"])
        writer.writerow(header)
        for i in range(n):
            row = [i]
            for metric in metrics:
                row.extend([int(results[metric].ranks[i]), float(results[metric].true_scores[i])])
            writer.writerow(row)


def save_retrieval_summary_csv(path: Path, summary: Dict[str, Dict[str, float]]) -> None:
    metric_names = list(summary.keys())
    fields = ["metric"] + [key for key in summary[metric_names[0]].keys()]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for metric in metric_names:
            row = {"metric": metric}
            row.update(summary[metric])
            writer.writerow(row)


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
    specs = [
        ("token_cos", "DINO token cosine (higher is better)"),
        ("dino_token_l1", "DINO token L1 (lower is better)"),
        ("dino_token_l2", "DINO token L2 (lower is better)"),
        ("dino_rel1_cos", "1st-order relation cosine (higher is better)"),
        ("dino_rel1_l1", "1st-order relation L1 (lower is better)"),
        ("dino_rel1_l2", "1st-order relation L2 (lower is better)"),
        ("dino_rel1_skl", "1st-order relation symmetric KL (lower is better)"),
        ("dino_rel2_cos", "2nd-order relation cosine (higher is better)"),
        ("dino_rel2_l1", "2nd-order relation L1 (lower is better)"),
        ("dino_rel2_l2", "2nd-order relation L2 (lower is better)"),
        ("dino_rel2_skl", "2nd-order relation symmetric KL (lower is better)"),
    ]
    specs = [(metric, title) for metric, title in specs if metric in paired_scores]
    cols = 3
    rows = int(np.ceil(len(specs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
    for ax, (metric, title) in zip(axes.flatten(), specs):
        ax.hist(paired_scores[metric], bins=24, alpha=0.65, label="paired")
        ax.hist(shuffled_scores[metric], bins=24, alpha=0.65, label="shuffled")
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.set_ylabel("count")
        ax.legend()
    for ax in axes.flatten()[len(specs) :]:
        ax.axis("off")
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
            f"true B rel={paired_scores['dino_rel1_cos'][r]:.3f}",
            f"shuf B rel={shuffled_scores['dino_rel1_cos'][r]:.3f}",
        ]
        for c, (img, label) in enumerate(zip(triplet, labels)):
            x = c * tile
            y = r * (tile + label_h)
            thumb = img.convert("RGB").resize((tile, tile), Image.Resampling.BICUBIC)
            canvas.paste(thumb, (x, y))
            draw_label(draw, (x + 4, y + tile + 4), label)
            if c == 1:
                draw_label(draw, (x + 4, y + tile + 22), f"2skl={paired_scores['dino_rel2_skl'][r]:.4f}")
            if c == 2:
                draw_label(draw, (x + 4, y + tile + 22), f"2skl={shuffled_scores['dino_rel2_skl'][r]:.4f}")

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


def run_full_retrieval(
    output_dir: Path,
    images_a: Sequence[Image.Image],
    images_b: Sequence[Image.Image],
    tokens_a: torch.Tensor,
    tokens_b: torch.Tensor,
    args: argparse.Namespace,
    ranking_device: torch.device,
) -> Dict[str, Dict[str, float]]:
    selected_metrics = parse_retrieval_metrics(args.retrieval_metrics)
    dtype = storage_dtype(args.storage_dtype)
    qbs = max(1, int(args.ranking_query_batch_size))
    cbs = max(1, int(args.ranking_candidate_batch_size))
    results: Dict[str, RetrievalResult] = {}

    pixel_metrics = {m for m in selected_metrics if m.startswith("pixel_")}
    if pixel_metrics:
        pixel_a = images_to_pixel_vectors(images_a, args.pixel_size, dtype)
        pixel_b = images_to_pixel_vectors(images_b, args.pixel_size, dtype)
        if "pixel_l1" in pixel_metrics:
            results["pixel_l1"] = rank_vector_metric("pixel_l1", pixel_a, pixel_b, "l1", False, ranking_device, qbs, cbs)
        if "pixel_l2" in pixel_metrics:
            results["pixel_l2"] = rank_vector_metric("pixel_l2", pixel_a, pixel_b, "l2", False, ranking_device, qbs, cbs)
        del pixel_a, pixel_b

    token_metrics = {m for m in selected_metrics if m.startswith("dino_token_")}
    if token_metrics:
        if {"dino_token_l1", "dino_token_l2"} & token_metrics:
            token_a = flatten_batches(tokens_a, dtype, args.batch_size, "Flattening DINO tokens A")
            token_b = flatten_batches(tokens_b, dtype, args.batch_size, "Flattening DINO tokens B")
            if "dino_token_l1" in token_metrics:
                results["dino_token_l1"] = rank_vector_metric(
                    "dino_token_l1", token_a, token_b, "l1", False, ranking_device, qbs, cbs
                )
            if "dino_token_l2" in token_metrics:
                results["dino_token_l2"] = rank_vector_metric(
                    "dino_token_l2", token_a, token_b, "l2", False, ranking_device, qbs, cbs
                )
            del token_a, token_b
        if "dino_token_cos" in token_metrics:
            token_cos_a = normalized_token_flat_batches(tokens_a, dtype, args.batch_size, "Normalizing DINO tokens A")
            token_cos_b = normalized_token_flat_batches(tokens_b, dtype, args.batch_size, "Normalizing DINO tokens B")
            results["dino_token_cos"] = rank_vector_metric(
                "dino_token_cos",
                token_cos_a,
                token_cos_b,
                "cos",
                True,
                ranking_device,
                qbs,
                cbs,
                scale=1.0 / float(tokens_a.shape[1]),
            )
            del token_cos_a, token_cos_b

    rel1_metrics = {m for m in selected_metrics if m.startswith("dino_rel1_")}
    rel2_metrics = {m for m in selected_metrics if m.startswith("dino_rel2_")}
    if rel1_metrics or rel2_metrics:
        rel1_a, rel2_a = build_relation_flats(tokens_a, args.remove_diag, dtype, args.batch_size, bool(rel2_metrics))
        rel1_b, rel2_b = build_relation_flats(tokens_b, args.remove_diag, dtype, args.batch_size, bool(rel2_metrics))
        row_count = infer_patch_grid(tokens_a.shape[1])[0]

        if "dino_rel1_l1" in rel1_metrics:
            results["dino_rel1_l1"] = rank_vector_metric("dino_rel1_l1", rel1_a, rel1_b, "l1", False, ranking_device, qbs, cbs)
        if "dino_rel1_l2" in rel1_metrics:
            results["dino_rel1_l2"] = rank_vector_metric("dino_rel1_l2", rel1_a, rel1_b, "l2", False, ranking_device, qbs, cbs)
        if "dino_rel1_cos" in rel1_metrics:
            rel1_cos_a = normalized_flat_batches(rel1_a, dtype, args.batch_size, "Normalizing relation-1 A")
            rel1_cos_b = normalized_flat_batches(rel1_b, dtype, args.batch_size, "Normalizing relation-1 B")
            results["dino_rel1_cos"] = rank_vector_metric(
                "dino_rel1_cos", rel1_cos_a, rel1_cos_b, "cos", True, ranking_device, qbs, cbs
            )
            del rel1_cos_a, rel1_cos_b
        if "dino_rel1_skl" in rel1_metrics:
            rel1_kl_a = make_kl_cache(rel1_a, row_count, args.tau, dtype, args.batch_size)
            rel1_kl_b = make_kl_cache(rel1_b, row_count, args.tau, dtype, args.batch_size)
            results["dino_rel1_skl"] = rank_skl_metric("dino_rel1_skl", rel1_kl_a, rel1_kl_b, ranking_device, qbs, cbs)
            del rel1_kl_a, rel1_kl_b

        if rel2_metrics:
            if rel2_a is None or rel2_b is None:
                raise RuntimeError("Internal error: second-order relation metrics requested but not built.")
            if "dino_rel2_l1" in rel2_metrics:
                results["dino_rel2_l1"] = rank_vector_metric("dino_rel2_l1", rel2_a, rel2_b, "l1", False, ranking_device, qbs, cbs)
            if "dino_rel2_l2" in rel2_metrics:
                results["dino_rel2_l2"] = rank_vector_metric("dino_rel2_l2", rel2_a, rel2_b, "l2", False, ranking_device, qbs, cbs)
            if "dino_rel2_cos" in rel2_metrics:
                rel2_cos_a = normalized_flat_batches(rel2_a, dtype, args.batch_size, "Normalizing relation-2 A")
                rel2_cos_b = normalized_flat_batches(rel2_b, dtype, args.batch_size, "Normalizing relation-2 B")
                results["dino_rel2_cos"] = rank_vector_metric(
                    "dino_rel2_cos", rel2_cos_a, rel2_cos_b, "cos", True, ranking_device, qbs, cbs
                )
                del rel2_cos_a, rel2_cos_b
            if "dino_rel2_skl" in rel2_metrics:
                rel2_kl_a = make_kl_cache(rel2_a, row_count, args.tau, dtype, args.batch_size)
                rel2_kl_b = make_kl_cache(rel2_b, row_count, args.tau, dtype, args.batch_size)
                results["dino_rel2_skl"] = rank_skl_metric("dino_rel2_skl", rel2_kl_a, rel2_kl_b, ranking_device, qbs, cbs)
                del rel2_kl_a, rel2_kl_b

        del rel1_a, rel1_b, rel2_a, rel2_b

    ordered_results = {metric: results[metric] for metric in selected_metrics}
    retrieval_summary = {
        metric: summarize_retrieval_result(result, candidate_count=len(images_b))
        for metric, result in ordered_results.items()
    }
    save_retrieval_ranks_csv(output_dir / "retrieval_ranks.csv", ordered_results)
    save_retrieval_summary_csv(output_dir / "retrieval_summary.csv", retrieval_summary)
    with (output_dir / "retrieval_summary.json").open("w") as f:
        json.dump(
            {
                "selected_metrics": selected_metrics,
                "candidate_count": len(images_b),
                "ranking_device": str(ranking_device),
                "ranking_query_batch_size": qbs,
                "ranking_candidate_batch_size": cbs,
                "pixel_size": args.pixel_size,
                "metric_token_grid": args.metric_token_grid,
                "storage_dtype": args.storage_dtype,
                "metrics": retrieval_summary,
            },
            f,
            indent=2,
        )
    return retrieval_summary


def main() -> None:
    args = parse_args()
    requested_split = args.split
    args.split = resolve_split(args.dataset, args.split)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    ranking_device = device if args.ranking_device == "auto" else choose_device(args.ranking_device)
    dtype = storage_dtype(args.storage_dtype)
    print(f"device={device}")
    print(f"ranking_device={ranking_device}")
    print(f"requested_split={requested_split} resolved_split={args.split}")
    print(f"dataset={args.dataset} split={args.split} columns=({args.column_a}, {args.column_b})")

    images_a, images_b = load_samples(args)
    shuffled_indices = derangement_indices(len(images_a), args.seed + 17)

    extractor = DinoExtractor(args, device)
    tokens_a, token_grid_a = extractor.extract(images_a, args.batch_size, args.metric_token_grid, dtype)
    tokens_b, token_grid_b = extractor.extract(images_b, args.batch_size, args.metric_token_grid, dtype)
    if token_grid_a != token_grid_b:
        raise RuntimeError(f"A/B DINO token grids differ: {token_grid_a} vs {token_grid_b}")

    summary = {
        "dataset": args.dataset,
        "requested_split": requested_split,
        "split": args.split,
        "column_a": args.column_a,
        "column_b": args.column_b,
        "num_samples": len(images_a),
        "model_repo": args.model_repo,
        "model_name": args.model_name,
        "image_size": args.image_size,
        "patch_grid": list(token_grid_a),
        "metric_token_grid": args.metric_token_grid,
        "token_subsample": args.token_subsample,
        "remove_diag": args.remove_diag,
        "overlay_anchors": args.overlay_anchors,
        "shuffled_indices_rule": "cyclic random shift derangement",
        "metrics": {},
        "retrieval": {},
    }

    if not args.skip_paired_shuffled:
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
            metric: metric == "token_cos" or metric.endswith("_cos")
            for metric in paired_scores.keys()
        }
        for metric, hib in higher_is_better.items():
            summary["metrics"][metric] = summarize_metric(
                paired_scores[metric],
                shuffled_scores[metric],
                higher_is_better=hib,
                reps=args.bootstrap_reps,
                seed=args.seed + 101,
            )

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

    if not args.skip_retrieval:
        summary["retrieval"] = run_full_retrieval(
            output_dir,
            images_a,
            images_b,
            tokens_a,
            tokens_b,
            args,
            ranking_device,
        )

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if summary["metrics"]:
        print(json.dumps(summary["metrics"], indent=2))
    if summary["retrieval"]:
        print(json.dumps(summary["retrieval"], indent=2))
    print(f"saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
