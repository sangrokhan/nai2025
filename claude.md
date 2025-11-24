# Cellular Network Optimization - Hierarchical Model

## Project Overview
셀룰러 네트워크의 throughput을 예측하고 최적화하기 위한 계층적 딥러닝 모델입니다.
- **Physical Layer Encoder**: CQI, SINR, RI 처리
- **Link Adaptation Encoder**: MCS 통계 처리
- **Auxiliary Tasks**: 각 계층의 표현 학습 품질 검증
- **Main Task**: Throughput 예측

## Project Structure
```
project/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── physical_encoder.py      # Physical Layer Encoder
│   │   ├── la_encoder.py             # Link Adaptation Encoder
│   │   ├── auxiliary_tasks.py        # Auxiliary task modules
│   │   └── hierarchical_model.py     # 통합 모델
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessor.py           # 데이터 전처리
│   │   └── dataset.py                # PyTorch Dataset
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py                 # Loss 함수들
│   │   ├── trainer.py                # 학습 루프
│   │   └── logger.py                 # TensorBoard 로거
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── analyzer.py               # 모델 분석
│   └── utils/
│       ├── __init__.py
│       └── early_stopping.py         # Early stopping
├── tests/
│   ├── test_models.py                # 모델 테스트
│   ├── test_data.py                  # 데이터 테스트
│   ├── test_training.py              # 학습 테스트
│   └── test_integration.py           # 통합 테스트
├── configs/
│   ├── small_model.yaml              # 작은 모델 설정
│   ├── medium_model.yaml             # 중간 모델 설정 (기본)
│   └── large_model.yaml              # 큰 모델 설정
├── scripts/
│   ├── train.py                      # 학습 스크립트
│   ├── evaluate.py                   # 평가 스크립트
│   └── analyze.py                    # 분석 스크립트
├── data/                             # 데이터 디렉토리
├── checkpoints/                      # 체크포인트 저장
├── runs/                             # TensorBoard 로그
├── analysis_output/                  # 분석 결과
├── requirements.txt
├── setup.py
└── README.md
```

## Key Implementation Details

### 1. Data Processing
- **Input Features**:
  - Physical Layer: CQI (16×4), SINR (20×2), RI (4) → 총 116 features
  - Link Adaptation: MCS (32×8) → 총 256 features
  - Context: UE count, PRB utilization
- **Preprocessing**:
  - log1p transformation
  - Within-group normalization
  - 각 feature group별 독립 처리

### 2. Model Architecture

#### Physical Layer Encoder
```
Input (116 features)
→ Feature Encoders (7개: CQI×4, SINR×2, RI×1)
→ Transformer (multi-head attention)
→ h_channel (128-dim)
```

#### Link Adaptation Encoder
```
Input (256 MCS features + h_channel)
→ MCS Encoders (8개: 4 layers × 2 MIMO types)
→ Channel-aware modulation (FiLM)
→ Transformer integration
→ h_LA (128-dim)
```

#### Throughput Predictor
```
[h_channel, h_LA, ue_count, prb_util]
→ Deep MLP (3-4 layers)
→ Throughput prediction (1-dim)
```

### 3. Loss Functions

#### Main Loss
- Throughput MSE (log space)
- Throughput MAE (original space)

#### Physical Auxiliary Losses
- Spectral Efficiency prediction
- RI distribution prediction (KL divergence)
- Channel quality prediction

#### LA Auxiliary Losses
- MCS average prediction
- SU/MU MIMO ratio prediction

**Total Loss**:
```
L_total = L_throughput + α·L_physical_aux + β·L_la_aux
```

### 4. Training Configuration

**Medium Model (Recommended)**:
- `channel_dim`: 128
- `num_transformer_layers`: 2
- `num_attention_heads`: 8
- `dropout`: 0.15
- `weight_decay`: 5e-5
- `learning_rate`: 5e-4 → 1e-4 (with scheduling)

**Current Performance**:
- R²: 0.7713
- MAPE: 10.90%
- Train/Val gap: ~11% (정상 범위)

## Implementation Tasks

### Phase 1: Core Implementation (Priority: High)

#### Task 1.1: Data Module
```python
# src/data/preprocessor.py
class CellularDataPreprocessor:
    """
    셀룰러 네트워크 데이터 전처리

    Requirements:
    - Physical layer groups: CQI (×4), SINR (×2), RI
    - LA layer groups: MCS (4 layers × 2 MIMO types × 32 indices)
    - log1p + normalization
    """

# src/data/dataset.py
class HierarchicalDataset(Dataset):
    """
    PyTorch Dataset

    Requirements:
    - Physical features 전처리
    - LA features 전처리
    - Target (throughput) log transformation
    - Context features (ue_count, prb_util)
    """
```

**MCS Data Access Pattern (중요!)**:
```python
# MCS 데이터는 다음 naming convention 사용:
for layer in ['ONE_LAYER', 'TWO_LAYER', 'THREE_LAYER', 'FOUR_LAYER']:
    for mimo_type in ['SU_MIMO', 'MU_MIMO']:
        # 실제 컬럼명: f'MCS_{layer}_{mimo_type}_MCS{i}' (i=0~31)
        # Dataset key: f'mcs_{layer.lower()}_{mimo_type.lower()}'

# 예시:
# - 'mcs_one_layer_su_mimo': [B, 32]
# - 'mcs_two_layer_mu_mimo': [B, 32]
```

#### Task 1.2: Model Modules
```python
# src/models/physical_encoder.py
class PhysicalLayerEncoder(nn.Module):
    """
    Physical Layer → h_channel

    Components:
    - 7 feature encoders (CQI×4, SINR×2, RI×1)
    - Multi-head attention (Transformer)
    - Aggregation MLP

    Output: h_channel [B, channel_dim]
    """

# src/models/la_encoder.py
class LinkAdaptationEncoder(nn.Module):
    """
    MCS + h_channel → h_LA

    Components:
    - 8 MCS encoders (4 layers × 2 MIMO)
    - Channel-aware modulation (FiLM)
    - Transformer integration
    - Fusion with h_channel

    Output: h_LA [B, channel_dim]
    """

# src/models/auxiliary_tasks.py
class AuxiliaryTasks(nn.Module):
    """Physical layer auxiliary tasks"""

class LAAuxiliaryTasks(nn.Module):
    """LA layer auxiliary tasks"""

# src/models/hierarchical_model.py
class HierarchicalModel(nn.Module):
    """
    통합 모델

    Forward:
    1. h_channel = physical_encoder(batch)
    2. h_LA = la_encoder(batch, h_channel)
    3. auxiliary tasks on both h_channel and h_LA
    4. throughput_pred from [h_channel, h_LA, context]
    """
```

#### Task 1.3: Loss Functions
```python
# src/training/losses.py

def compute_auxiliary_losses(aux_outputs, batch, weights) -> Dict:
    """
    Physical auxiliary losses

    1. SE prediction: from SINR average
    2. RI distribution: KL divergence
    3. Channel quality: from CQI weighted average
    """

def compute_la_auxiliary_losses(la_aux_outputs, batch, weights) -> Dict:
    """
    LA auxiliary losses

    1. MCS average: weighted average across all MCS stats
    2. SU ratio: SU_total / (SU_total + MU_total)
    """

class HierarchicalLoss(nn.Module):
    """
    Total loss = main + physical_aux + la_aux

    IMPORTANT: Auxiliary losses가 0이 되지 않도록 주의!
    - 제대로 계산되는지 확인
    - loss dict에 제대로 포함되는지 확인
    """
```

#### Task 1.4: Training Loop
```python
# src/training/trainer.py
class Trainer:
    """
    학습 루프

    Features:
    - TensorBoard logging
    - Early stopping
    - Learning rate scheduling
    - Gradient clipping
    - Debug mode (첫 epoch에 상세 출력)

    Logging:
    - Train/Val for all loss components
    - R² and MAPE metrics
    - Learning rate
    - Model gradients & weights (optional)
    """

# src/training/logger.py
class TensorBoardLogger:
    """
    TensorBoard 로거

    Log:
    - Scalars: losses, metrics, lr
    - Histograms: gradients, weights
    - Embeddings: h_channel, h_LA
    """
```

### Phase 2: Testing (Priority: High)

#### Test 1: Data Pipeline
```python
# tests/test_data.py

def test_preprocessor():
    """
    Preprocessor 테스트
    - Physical groups 정확히 전처리되는지
    - LA groups (MCS) 정확히 전처리되는지
    - Normalization 올바른지
    """

def test_dataset():
    """
    Dataset 테스트
    - 올바른 shape 반환하는지
    - 모든 key가 존재하는지
    - Batch collation 동작하는지
    """

def test_dataloader():
    """
    DataLoader 테스트
    - 배치 생성 가능한지
    - 여러 worker에서 동작하는지
    """
```

#### Test 2: Model Components
```python
# tests/test_models.py

def test_physical_encoder():
    """
    Physical encoder 테스트
    - Forward pass 성공하는지
    - Output shape 맞는지
    - Gradient flow 되는지
    """

def test_la_encoder():
    """
    LA encoder 테스트
    - h_channel conditioning 작동하는지
    - MCS 모든 key 처리하는지
    - Output shape 맞는지
    """

def test_auxiliary_tasks():
    """
    Auxiliary tasks 테스트
    - 모든 prediction head 작동하는지
    - Output range 적절한지 ([0,1] 등)
    """

def test_hierarchical_model():
    """
    통합 모델 테스트
    - End-to-end forward pass
    - 모든 출력 생성되는지
    - Parameter count 예상 범위인지
    """
```

#### Test 3: Loss Functions
```python
# tests/test_training.py

def test_auxiliary_loss_computation():
    """
    Auxiliary loss 계산 테스트

    CRITICAL: 이게 제일 중요!
    - Physical aux loss가 0이 아닌지
    - LA aux loss가 0이 아닌지
    - 각 component별 loss 값 확인
    """

def test_hierarchical_loss():
    """
    Total loss 계산 테스트
    - 모든 component가 포함되는지
    - Weight 적용 올바른지
    - Backward 동작하는지
    """

def test_training_step():
    """
    한 스텝 학습 테스트
    - Forward → Loss → Backward → Update
    - Loss 감소하는지
    - Gradient 존재하는지
    """
```

#### Test 4: Integration
```python
# tests/test_integration.py

def test_full_training_loop():
    """
    소규모 데이터로 전체 학습 테스트
    - 2-3 epoch 학습
    - Loss 감소하는지
    - Checkpoint 저장되는지
    - TensorBoard 로그 생성되는지
    """

def test_overfitting_small_batch():
    """
    Overfitting 테스트 (모델 capacity 확인)
    - 작은 배치 (32 samples)
    - Train loss → 0에 가까워지는지
    - 모델이 데이터를 외울 수 있는지
    """
```

### Phase 3: Analysis Tools (Priority: Medium)

```python
# src/analysis/analyzer.py
class HierarchicalModelAnalyzer:
    """
    모델 분석 도구

    Features:
    1. Representation extraction (h_channel, h_LA)
    2. Clustering analysis
    3. Correlation with throughput
    4. Auxiliary task performance
    5. Prediction quality metrics
    6. Layer contribution analysis

    Outputs:
    - PNG plots (5-6개)
    - Summary statistics
    """
```

### Phase 4: Utilities (Priority: Low)

```python
# src/utils/early_stopping.py
class EarlyStopping:
    """Early stopping with patience"""

# src/utils/config.py
def load_config(yaml_path: str) -> Dict:
    """YAML config loader"""

# src/utils/metrics.py
def compute_metrics(y_true, y_pred) -> Dict:
    """R², MAPE, MAE, RMSE 계산"""
```

## Configuration Files

### configs/medium_model.yaml
```yaml
# 기본 설정 (추천)
model:
  channel_dim: 128
  num_transformer_layers: 2
  num_attention_heads: 8
  predictor_hidden_dim: 256
  predictor_num_layers: 3
  dropout: 0.15

training:
  batch_size: 128
  num_epochs: 100
  learning_rate: 5.0e-4
  weight_decay: 5.0e-5
  gradient_clip_norm: 1.0

loss:
  main_weight: 1.0
  physical_aux_weights:
    se: 0.25
    ri_dist: 0.15
    quality: 0.15
  la_aux_weights:
    mcs_avg: 0.15
    su_ratio: 0.08

scheduler:
  type: ReduceLROnPlateau
  mode: min
  factor: 0.5
  patience: 5
  min_lr: 1.0e-6

early_stopping:
  patience: 15
  min_delta: 1.0e-4

paths:
  train_data: ./data/train.parquet
  val_data: ./data/val.parquet
  save_dir: ./checkpoints
  log_dir: ./runs/medium_model
```

## Execution Scripts

### scripts/train.py
```python
"""
Main training script

Usage:
    python scripts/train.py --config configs/medium_model.yaml
    python scripts/train.py --config configs/medium_model.yaml --debug
"""

def main(config_path: str, debug: bool = False):
    # 1. Load config
    # 2. Setup data
    # 3. Create model
    # 4. Setup training
    # 5. Train
    # 6. Save results
```

### scripts/evaluate.py
```python
"""
Evaluation script

Usage:
    python scripts/evaluate.py \
        --checkpoint ./checkpoints/best_model.pt \
        --data ./data/test.parquet \
        --output ./evaluation_results.json
"""
```

### scripts/analyze.py
```python
"""
Analysis script

Usage:
    python scripts/analyze.py \
        --checkpoint ./checkpoints/best_model.pt \
        --data ./data/val.parquet \
        --output_dir ./analysis_output
"""
```

## Testing Commands

```bash
# 전체 테스트 실행
pytest tests/ -v

# 특정 테스트만 실행
pytest tests/test_data.py -v
pytest tests/test_models.py -v
pytest tests/test_training.py::test_auxiliary_loss_computation -v

# Coverage와 함께 실행
pytest tests/ --cov=src --cov-report=html

# 통합 테스트 (시간 오래 걸림)
pytest tests/test_integration.py -v -s
```

## Debug Mode

학습 시작 전 debug mode로 문제 확인:

```python
# scripts/train.py에서
if args.debug:
    # 1. 모델 forward pass 테스트
    # 2. Loss 계산 테스트
    # 3. 첫 배치 상세 출력
    # 4. Auxiliary loss 0이 아닌지 확인
```

**Debug checklist**:
- [ ] Forward pass 성공
- [ ] 모든 auxiliary outputs 존재
- [ ] Physical aux loss > 0
- [ ] LA aux loss > 0
- [ ] Total loss 올바르게 합쳐짐
- [ ] Backward 성공
- [ ] Gradients 존재

## Expected Performance

### Current (Medium Model, channel_dim=128)
```
Validation:
  R²: 0.7713
  MAPE: 10.90%
  Train Loss: 0.0455
  Val Loss: 0.0658
  Gap: 44.6% (total), 11% (throughput only)

Status: ✅ Production Ready
```

### Target (with improvements)
```
With regularization + augmentation:
  R²: 0.80-0.82
  MAPE: 9-10%
  Gap: <10% (throughput)

With ensemble:
  R²: 0.82-0.85
  MAPE: 8-9%
```

## Critical Implementation Notes

### 1. MCS Data Naming (매우 중요!)
```python
# 실제 CSV/Parquet 컬럼명
MCS_ONE_LAYER_SU_MIMO_MCS0
MCS_ONE_LAYER_SU_MIMO_MCS1
...
MCS_FOUR_LAYER_MU_MIMO_MCS31

# Preprocessor에서 group 생성 시
for layer in ['ONE_LAYER', 'TWO_LAYER', 'THREE_LAYER', 'FOUR_LAYER']:
    for mimo_type in ['SU_MIMO', 'MU_MIMO']:
        group_name = f'mcs_{layer.lower()}_{mimo_type.lower()}'

# Dataset __getitem__에서 반환 시
'mcs_one_layer_su_mimo': tensor([32])

# Model에서 접근 시
batch[f'mcs_{layer}_su_mimo']
```

### 2. Auxiliary Loss가 0이 되는 문제
**원인**:
- Loss 함수가 제대로 호출 안 됨
- Outputs dict에 auxiliary 결과 없음
- Loss dict에 합쳐지지 않음

**해결**:
- Debug mode로 첫 배치 확인
- 모든 auxiliary output 존재 확인
- Loss 계산 단계별 print

### 3. Regularization Strategy
```python
# 현재 gap이 있지만 성능은 좋음
# → Capacity 줄이지 말고 regularization 강화

dropout: 0.15  # 0.1 → 0.15
weight_decay: 5e-5  # 1e-5 → 5e-5
aux_weights: 0.7x  # 전체적으로 30% 감소
```

## Requirements

### Python Packages
```txt
torch>=2.0.0
tensorboard>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
pyarrow>=12.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pytest>=7.4.0
pytest-cov>=4.1.0
pyyaml>=6.0
```

### Hardware
- GPU: 8GB+ VRAM (RTX 3070 이상 권장)
- RAM: 16GB+
- Storage: 10GB+ (데이터 + 체크포인트)

## Getting Started

```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. 테스트 실행 (데이터 없이도 가능)
pytest tests/test_models.py -v

# 3. Debug mode로 학습 시작
python scripts/train.py --config configs/medium_model.yaml --debug

# 4. TensorBoard 확인
tensorboard --logdir=./runs

# 5. 정상 학습 시작
python scripts/train.py --config configs/medium_model.yaml

# 6. 분석 실행
python scripts/analyze.py \
    --checkpoint ./checkpoints/best_model.pt \
    --data ./data/val.parquet \
    --output_dir ./analysis_output
```

## Troubleshooting

### Issue 1: Auxiliary Loss = 0
```bash
# Debug mode 실행
python scripts/train.py --config configs/medium_model.yaml --debug

# 확인할 것:
# - "=== DEBUG: Model Outputs ===" 섹션
# - physical_aux, la_aux 출력 존재하는지
# - "=== DEBUG: Loss Values ===" 섹션
# - physical_total_aux, la_total_la_aux가 0이 아닌지
```

### Issue 2: Out of Memory
```yaml
# configs/medium_model.yaml 수정
training:
  batch_size: 64  # 128 → 64
  gradient_accumulation_steps: 2  # 추가
```

### Issue 3: Train/Val Gap Too Large
```yaml
# regularization 강화
model:
  dropout: 0.20  # 0.15 → 0.20

training:
  weight_decay: 1.0e-4  # 5e-5 → 1e-4
```

## Next Steps

1. **Data augmentation**: ImprovedAugmentation 클래스 구현
2. **Ensemble**: 여러 체크포인트 평균
3. **Hyperparameter tuning**: Optuna 사용
4. **More data**: 데이터 추가 수집 (성능 향상의 핵심)

---

## Important Notes for Claude Code

1. **Start with tests**: 구현 전에 테스트 먼저 작성
2. **Debug mode first**: 학습 전에 반드시 debug mode로 확인
3. **Modular design**: 각 component를 독립적으로 테스트 가능하게
4. **Clear naming**: MCS 데이터 naming convention 엄격히 준수
5. **Logging**: 모든 중요한 값들을 TensorBoard에 로깅

**Critical Path**:
1. Data pipeline 구현 및 테스트
2. Model components 구현 및 테스트
3. Loss functions 구현 및 테스트 (auxiliary loss 0 아닌지 확인!)
4. Training loop 구현
5. Debug mode로 전체 flow 확인
6. 실제 학습 시작

Good luck! 🚀
