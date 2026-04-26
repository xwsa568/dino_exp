# DINO Night2Day Pair Probe

Hugging Face `huggan/night2day`의 `imageA` / `imageB`가 실제 night/day 대응쌍이면, 같은 row의 DINO relation이 row를 섞은 negative pair보다 더 비슷해야 한다는 가정을 테스트하는 작은 standalone 실험입니다.

HF dataset viewer 기준 기본 split은 `train`, row 수는 20,120개이고 컬럼은 `imageA`, `imageB`입니다.

## Setup

```bash
cd experiments/dino_night2day_pair_probe
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_dino_pair_probe.py \
  --num-samples 128 \
  --batch-size 16 \
  --output-dir outputs/night2day_128
```

기본값은 `facebookresearch/dinov2`의 `dinov2_vitb14`를 `torch.hub`로 로드합니다. GPU가 있으면 `cuda`, Apple Silicon이면 `mps`, 아니면 `cpu`를 자동 선택합니다.

## Outputs

- `summary.json`: paired vs shuffled metric summary.
- `scores.csv`: sample별 paired/shuffled 점수.
- `score_distributions.png`: paired/shuffled 점수 분포.
- `pair_grid.png`: `imageA`, true `imageB`, shuffled `imageB` 비교 grid.
- `relation_heatmaps.png`: DINO patch-token self-similarity matrix와 차이 heatmap.

## Metrics

- `dino_rel_cos`: DINO patch-token self-similarity matrix 간 cosine similarity. 높을수록 비슷합니다.
- `dino_rel_mse`: self-similarity matrix MSE. 낮을수록 비슷합니다.
- `dino_rel_2gram_skl`: repo의 DINO relation loss와 같은 계열의 second-order relation KL을 symmetric하게 계산한 값. 낮을수록 비슷합니다.
- `token_cos`: 같은 spatial token 위치끼리의 평균 DINO cosine similarity. relation 전의 참고값입니다.

요약의 `improvement_mean`은 paired가 shuffled보다 좋은 방향을 양수로 정규화한 값입니다.
