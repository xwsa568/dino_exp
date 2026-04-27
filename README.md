# DINO Night2Day Pair Probe

Hugging Face `huggan/night2day`의 `imageA` / `imageB`가 실제 night/day 대응쌍이면, 같은 row의 DINO feature/relation이 전체 `imageB` 후보 중에서도 높은 순위로 매칭되어야 한다는 가정을 테스트하는 standalone 실험입니다.

`--split auto`는 dataset에 `test` split이 있으면 test를 쓰고, 없으면 `validation`, `train` 순서로 선택합니다. `--num-samples 0`은 선택된 split 전체를 평가합니다.

## Setup

```bash
cd /Users/xwsa568/Projects/dino_exp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 run_dino_pair_probe.py \
  --split auto \
  --num-samples 0 \
  --batch-size 16 \
  --metric-token-grid 8 \
  --output-dir outputs/night2day_full_retrieval
```

기본값은 `facebookresearch/dinov2`의 `dinov2_vitb14`를 `torch.hub`로 로드합니다. GPU가 있으면 `cuda`, Apple Silicon이면 `mps`, 아니면 `cpu`를 자동 선택합니다. 전체 split retrieval은 모든 A에 대해 모든 B를 후보로 비교하므로 오래 걸릴 수 있습니다. 기본 `--metric-token-grid 8`은 16x16 DINO patch token을 8x8로 줄여 relation metric 비용을 낮춥니다. 전체 patch token을 쓰려면 `--metric-token-grid 0`을 지정합니다.

## Outputs

- `summary.json`: paired-vs-shuffled summary와 full retrieval summary.
- `retrieval_summary.json`: metric별 true pair rank 요약.
- `retrieval_summary.csv`: `retrieval_summary.json`의 table 형태.
- `retrieval_ranks.csv`: sample별 true `imageB`의 rank와 true-pair score.
- `scores.csv`: sample별 paired/shuffled 점수.
- `score_distributions.png`: paired/shuffled 점수 분포.
- `pair_grid.png`: `imageA`, true `imageB`, shuffled `imageB` 비교 grid.
- `relation_heatmaps.png`: DINO patch-token self-similarity matrix와 차이 heatmap.
- `patch_relation_overlays.png`: 선택한 anchor patch가 나머지 patch들에 대해 갖는 relation row를 실제 A/B 이미지 위에 overlay한 그림.
- `patch_relation_similarity_maps.png`: 모든 patch를 anchor로 봤을 때 A/B의 relation row가 얼마나 비슷한지 이미지 위에 overlay한 그림.

## Metrics

- `pixel_l1`, `pixel_l2`: resized RGB pixel vector 차이. 낮을수록 비슷합니다.
- `dino_token_cos`, `dino_token_l1`, `dino_token_l2`: 같은 spatial DINO patch feature끼리 직접 비교한 metric입니다.
- `dino_rel1_cos`, `dino_rel1_l1`, `dino_rel1_l2`, `dino_rel1_skl`: DINO patch-token self-similarity matrix, 즉 1차 relation matrix 비교입니다.
- `dino_rel2_cos`, `dino_rel2_l1`, `dino_rel2_l2`, `dino_rel2_skl`: 1차 relation row들 사이의 relation, 즉 2차 relation matrix 비교입니다.
- `*_skl`: row-wise softmax 후 symmetric KL을 평균낸 값입니다. 낮을수록 비슷합니다.

Paired-vs-shuffled 요약의 `improvement_mean`은 paired가 shuffled보다 좋은 방향을 양수로 정규화한 값입니다. Retrieval 요약에서는 `mean_rank`, `median_rank`, `mrr`, `top1_rate`, `top5_rate`, `top10_rate`, `top1pct_rate`, `mean_pair_percentile`로 어떤 metric이 true pair를 가장 잘 찾는지 비교합니다.

## Patch Relation Overlays

`patch_relation_overlays.png`는 DINO relation matrix `N x N`에서 특정 anchor patch의 row 하나를 `H_patch x W_patch`로 reshape해서 원본 이미지 위에 올립니다. 같은 위치의 anchor를 A/B에 모두 사용하므로, 두 overlay가 비슷하면 해당 위치 patch가 이미지 전체에 대해 갖는 구조적 관계가 night/day pair 사이에서 비슷하다는 뜻입니다.

기본 anchor는 `center,upper_left,upper_right,lower_left,lower_right`입니다. 바꾸려면:

```bash
python run_dino_pair_probe.py \
  --overlay-anchors center,4:4,8:8 \
  --num-overlay-samples 4
```

`patch_relation_similarity_maps.png`는 모든 patch를 anchor로 삼아 relation row cosine/MSE를 계산한 spatial map입니다. 밝은/high cosine 영역은 같은 위치 patch의 relation pattern이 pair 사이에서 잘 맞는 부분입니다.
