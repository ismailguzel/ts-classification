# Hierarchical Time Series Classification Models

Bu klasör, hiyerarşik zaman serisi sınıflandırma sistemi için eğitim scriptlerini içerir.

##  Klasör Yapısı

```
03-models/hierarchical/
├── model1_binary/              # Binary classification (Stationary vs Non-Stationary)
│   ├── train_model1.py        # Training script
│   ├── test_model1.py         # Test script
│   ├── slurm_train_model1.sh  # Example SLURM script
│   └── README.md              # Model 1 documentation
│
├── model2_nonstationary/       # 5-class classification (Non-Stationary types)
│   ├── train_model2.py        # Training script
│   ├── test_model2.py         # Test script
│   ├── slurm_train_model2.sh  # Example SLURM script
│   └── README.md              # Model 2 documentation
│
└── submit_training.sh          # Interactive job submission helper
```

##  Hiyerarşik Sistem

```
                    ┌─────────────────┐
                    │  Input Series   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    MODEL 1      │
                    │  (Binary Class) │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
      ┌───────▼────────┐          ┌────────▼────────┐
      │  STATIONARY    │          │ NON-STATIONARY  │
      │   (Class 0)    │          │   (Class 1)     │
      └────────────────┘          └────────┬────────┘
                                           │
                                  ┌────────▼────────┐
                                  │    MODEL 2      │
                                  │  (5-Class)      │
                                  └────────┬────────┘
                                           │
                        ┌──────────────────┼──────────────────┐
                        │         │        │        │         │
                   ┌────▼───┐ ┌──▼──┐ ┌──▼───┐ ┌──▼───┐ ┌──▼─────┐
                   │ Trend  │ │Volat│ │Stoch │ │Anomly│ │Struct  │
                   │(Class0)│ │(Cl1)│ │(Cl2) │ │(Cl3) │ │(Class4)│
                   └────────┘ └─────┘ └──────┘ └──────┘ └────────┘
```

## Hızlı Başlangıç

### Lokal Ortamda (Test/Development)

```bash
# Model 1 eğitimi
cd model1_binary
python train_model1.py --mode raw --classifier minirocket

# Model 2 eğitimi
cd model2_nonstationary
python train_model2.py --mode raw --classifier minirocket
```

## Model Özellikleri

### Model 1: Binary Classification

| Metrik | Değer |
|--------|-------|
| **Task** | Stationary vs Non-Stationary |
| **Classes** | 2 (Binary) |
| **Task** | Stationary vs Non-Stationary |

**Kullanılabilir Classifiers**:
- TimeSeriesForest (baseline, fast)
- ROCKET (balanced approach)
- Arsenal (ensemble)

### Model 2: 5-Class Classification

| Metrik | Değer |
|--------|-------|
| **Task** | Non-Stationary Type Classification |
| **Classes** | 5 (Trend/Volatility/Stochastic/Anomaly/Structural) |
| **Task** | Non-Stationary Type Classification |

**5 Classes**:
0. **Trend**: Deterministic trend patterns
1. **Volatility**: Changing variance
2. **Stochastic**: Random walk behavior
3. **Anomaly**: Point and collective anomalies
4. **Structural Break**: Sudden regime changes

**Kullanılabilir Classifiers**:
- TimeSeriesForest (baseline, fast)
- ROCKET (2000 kernels for 5-class)
- Arsenal (2000 kernels)

## Training Modes

Her iki model de **dual-mode** desteği sunar:

### 1. RAW Mode (Default, Önerilen)

Raw time series ile çalışır. sktime classifiers kullanır.

```bash
python train_model1.py --mode raw --classifier rocket
```

**Avantajlar**:
- Feature engineering gerektirmez
- Direkt zaman serisi üzerinde çalışır
- sktime'ın güçlü classifiers'ı
- Daha az preprocessing

**Available Classifiers**: `tsf`, `rocket`, `arsenal`, `all`

### 2. FEATURES Mode (Optional)

TSFresh features ile çalışır. sklearn classifiers kullanır.

```bash
# Önce features extract edin
cd ../../02-preprocessing
python extract_features.py

# Sonra FEATURES mode ile eğitin
cd ../03-models/hierarchical/model1_binary
python train_model1.py --mode features --features-path ../../../data/features/selected
```

**Avantajlar**:
- Traditional ML (RandomForest, XGBoost, SVM)
- Feature importance analysis
- Daha hızlı inference (features pre-computed)

### 3. AutoTrain Mode (AutoML)

Otomatik model seçimi ve hiperparametre optimizasyonu için AutoML framework'leri kullanır.

```bash
# AutoGluon ile
cd model1_binary
python autotrain_models1.py --engine autogluon \
    --features-path ../../../data/features/selected \
    --time-limit 3600 --presets medium_quality_faster_train

# PyCaret ile
python autotrain_models1.py --engine pycaret \
    --features-path ../../../data/features/selected \
    --folds 5
```

**Desteklenen Engines**:
- `autogluon`: AutoGluon Tabular (ensemble + stacking)
- `pycaret`: PyCaret Classification (20+ model karşılaştırma)

**Avantajlar**:
- Otomatik model seçimi ve hyperparameter tuning
- Ensemble ve stacking modelleri
- Minimal manual tuning gereksinimi
- Production-ready model artifacts

**Gereksinimler**:
- Önceden extract edilmiş features (FEATURES mode preprocessing gerekli)
- AutoGluon veya PyCaret kurulu olmalı

<!-- Performance comparison removed to keep documentation usage-focused. -->

## Kullanım Önerileri

### Test/Prototype İçin

```bash
# Hızlı test için TimeSeriesForest veya ROCKET kullanın
python train_model1.py --mode raw --classifier tsf
python train_model2.py --mode raw --classifier tsf
```

### Production İçin

```bash
# En iyi doğruluk için Arsenal kullanın
python train_model1.py --mode raw --classifier arsenal
python train_model2.py --mode raw --classifier arsenal
```

### Tüm Classifiers'ı Karşılaştırmak İçin

```bash
# Hepsini eğitin ve en iyisini seçin
python train_model1.py --mode raw --classifier all
python train_model2.py --mode raw --classifier all
```

### AutoML ile Hızlı Baseline

```bash
# AutoGluon ile otomatik model seçimi
cd model1_binary
python autotrain_models1.py --engine autogluon \
    --features-path ../../../data/features/selected

cd ../model2_nonstationary
python autotrain_models2.py --engine autogluon \
    --features-path ../../../data/features/selected
```

## Testing Models

Eğitilen modelleri test dataseti üzerinde değerlendirin:

```bash
# Test Model 1 (RAW mode)
cd model1_binary
python test_model1.py \
    --model-path saved_models/model1_binary_raw_rocket/model1_binary_classifier.pkl \
    --data-path ../../../data/raw/unified-test \
    --output-path results/

# Test Model 1 (FEATURES mode)
python test_model1.py \
    --model-path saved_models/model1_binary_features/model1_binary_classifier.pkl \
    --data-path ../../../data/raw/unified-test \
    --output-path results/

# Test Model 2 (RAW mode)
cd ../model2_nonstationary
python test_model2.py \
    --model-path saved_models/model2_nonstationary_raw_rocket/model2_nonstationary_classifier.pkl \
    --data-path ../../../data/raw/unified-test \
    --output-path results/

# Test Model 2 (FEATURES mode)
python test_model2.py \
    --model-path saved_models/model2_nonstationary_features/model2_nonstationary_classifier.pkl \
    --data-path ../../../data/raw/unified-test \
    --output-path results/
```

Test scriptleri şunları hesaplar:
- Evaluation metrics
- Confusion matrix
- Per-class metrics
- Classification report

##  Daha Fazla Bilgi

- **Model 1 Detayları**: `model1_binary/README.md`
- **Model 2 Detayları**: `model2_nonstationary/README.md`

##  Workflow

```bash
# 1. Veri generation (01-data-generation/)
python generate.py

# 2. Feature extraction (opsiyonel, 02-feature-engineering/)
python extract_features.py

# 3. Model training (03-models/hierarchical/)
python train_model1.py --mode raw
python train_model2.py --mode raw

# 4. Testing (03-models/hierarchical/)
python test_model1.py --model-path models/model1_arsenal_raw.pkl
python test_model2.py --model-path models/model2_arsenal_raw.pkl
```

##  Önemli Notlar

1. **Veri seti gerekli**: Training yapmadan önce `01-data-generation/` ile veri oluşturun
2. **HIVECOTEV2 çok yavaş**: Bu classifier sadece research/benchmark için kullanın
3. **Memory kullanımı**: Model 2, 5-class olduğu için daha fazla RAM kullanır
4. **Parallel training**: Model 1 ve Model 2'yi paralel çalıştırabilirsiniz (bağımsızlar)

## 🆘 Sorun Giderme

### "Dataset not found" hatası

```bash
# Veri setini oluşturun
cd ../../01-data-generation
python generate.py
```

### Out of memory hatası

```bash
# Test-size'ı artırın (daha az training data)
python train_model1.py --test-size 0.3  # varsayılan 0.2

# Veya daha küçük veri seti kullanın
python train_model1.py --data-path ../../../data/raw/unified-test
```

### sktime import hatası

```bash
# sktime'ı güncelleyin
pip install --upgrade sktime

# Veya problemsiz classifier kullanın
python train_model1.py --classifier minirocket
```

---

**Hazırlayan**: GitHub Copilot  
**Son Güncelleme**: 31 Ekim 2025  
**Python**: >= 3.9  
**sktime**: >= 0.24.0  
**scikit-learn**: >= 1.3.0
