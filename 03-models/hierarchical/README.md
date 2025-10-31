# Hierarchical Time Series Classification Models

Bu klasör, hiyerarşik zaman serisi sınıflandırma sistemi için eğitim scriptlerini içerir.

## 📁 Klasör Yapısı

```
03-models/hierarchical/
├── model1_binary/              # Binary classification (Stationary vs Non-Stationary)
│   ├── train_model1.py        # Training script
│   ├── test_model1.py         # Test script
│   ├── slurm_train_model1.sh  # TRUBA SLURM script
│   └── README.md              # Model 1 documentation
│
├── model2_nonstationary/       # 5-class classification (Non-Stationary types)
│   ├── train_model2.py        # Training script
│   ├── test_model2.py         # Test script
│   ├── slurm_train_model2.sh  # TRUBA SLURM script
│   └── README.md              # Model 2 documentation
│
├── submit_training.sh          # Interactive job submission helper
└── TRUBA_TRAINING_GUIDE.md    # Complete TRUBA guide
```

## 🎯 Hiyerarşik Sistem

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

## 🚀 Hızlı Başlangıç

### Lokal Ortamda (Test/Development)

```bash
# Model 1 eğitimi
cd model1_binary
python train_model1.py --mode raw --classifier minirocket

# Model 2 eğitimi
cd model2_nonstationary
python train_model2.py --mode raw --classifier minirocket
```

### TRUBA Cluster (Production)

**En kolay yol - İnteraktif helper kullanın**:

```bash
cd /arf/home/iguzel/ts-stationary/hierarchical-ts-classification/03-models/hierarchical
bash submit_training.sh
```

Script size şunları sorar:
1. Hangi model? (Model 1, Model 2, veya her ikisi)
2. Hangi classifier? (MiniROCKET ⭐, Arsenal ⭐, vs.)

Ardından otomatik olarak job'ları submit eder.

## 📊 Model Özellikleri

### Model 1: Binary Classification

| Metrik | Değer |
|--------|-------|
| **Task** | Stationary vs Non-Stationary |
| **Classes** | 2 (Binary) |
| **Accuracy** | 93-98% |
| **Training Time** | 20 dk - 5 saat (classifier'a göre) |
| **Recommended** | MiniROCKET ⭐ (hız) veya Arsenal ⭐ (doğruluk) |

**Kullanılabilir Classifiers**:
- TimeSeriesForest (baseline)
- ROCKET (good balance)
- MiniROCKET ⭐ (10x faster than ROCKET)
- Arsenal ⭐ (ensemble, best accuracy)
- ShapeletTransform (interpretable)
- HIVECOTEV2 (most powerful, very slow)

### Model 2: 5-Class Classification

| Metrik | Değer |
|--------|-------|
| **Task** | Non-Stationary Type Classification |
| **Classes** | 5 (Trend/Volatility/Stochastic/Anomaly/Structural) |
| **Accuracy** | 80-92% |
| **Training Time** | 30 dk - 10 saat (classifier'a göre) |
| **Recommended** | MiniROCKET ⭐ (hız) veya Arsenal ⭐ (doğruluk) |

**5 Classes**:
0. **Trend**: Deterministic trend patterns
1. **Volatility**: Changing variance
2. **Stochastic**: Random walk behavior
3. **Anomaly**: Point and collective anomalies
4. **Structural Break**: Sudden regime changes

**Kullanılabilir Classifiers**:
- TimeSeriesForest (baseline)
- ROCKET (2000 kernels for 5-class)
- MiniROCKET ⭐ (fastest)
- Arsenal ⭐ (2000 kernels, best accuracy)
- ShapeletTransform (interpretable)
- HIVECOTEV2 (most powerful, very slow)

## 🎛️ Training Modes

Her iki model de **dual-mode** desteği sunar:

### 1. RAW Mode (Default, Önerilen)

Raw time series ile çalışır. sktime classifiers kullanır.

```bash
python train_model1.py --mode raw --classifier minirocket
```

**Avantajlar**:
- ✅ Feature engineering gerektirmez
- ✅ Direkt zaman serisi üzerinde çalışır
- ✅ sktime'ın güçlü classifiers'ı
- ✅ Daha az preprocessing

### 2. FEATURES Mode (Optional)

TSFresh features ile çalışır. sklearn classifiers kullanır.

```bash
# Önce features extract edin
cd ../../02-feature-engineering
python extract_features.py

# Sonra FEATURES mode ile eğitin
cd ../03-models/hierarchical/model1_binary
python train_model1.py --mode features --features-path ../../../data/features/selected
```

**Avantajlar**:
- ✅ Traditional ML (RandomForest, XGBoost, SVM)
- ✅ Feature importance analysis
- ✅ Daha hızlı inference (features pre-computed)

## 📈 Performans Karşılaştırması

### Hız vs Doğruluk (Model 1, 150K dataset)

| Classifier | Accuracy | Training Time | Speed Rating | Recommendation |
|-----------|----------|---------------|--------------|----------------|
| MiniROCKET | 95-97% | ~25 min | ⚡⚡⚡⚡⚡ | ⭐ **Hız için en iyi** |
| ROCKET | 95-97% | ~1 hour | ⚡⚡⚡⚡ | Good balance |
| Arsenal | 96-98% | ~1.5 hour | ⚡⚡⚡ | ⭐ **Doğruluk için en iyi** |
| TimeSeriesForest | 93-95% | ~40 min | ⚡⚡⚡⚡ | Baseline |
| Shapelet | 93-96% | ~2 hour | ⚡⚡ | Interpretable |
| HIVECOTEV2 | 97-99% | ~5 hour | ⚡ | Most powerful |

### Hız vs Doğruluk (Model 2, 150K non-stationary samples)

| Classifier | Accuracy | Training Time | Speed Rating | Recommendation |
|-----------|----------|---------------|--------------|----------------|
| MiniROCKET | 84-87% | ~45 min | ⚡⚡⚡⚡⚡ | ⭐ **Hız için en iyi** |
| ROCKET | 85-88% | ~2 hour | ⚡⚡⚡⚡ | Good balance |
| Arsenal | 86-90% | ~2.5 hour | ⚡⚡⚡ | ⭐ **Doğruluk için en iyi** |
| TimeSeriesForest | 80-85% | ~1 hour | ⚡⚡⚡⚡ | Baseline |
| Shapelet | 82-86% | ~3 hour | ⚡⚡ | Interpretable |
| HIVECOTEV2 | 88-92% | ~10 hour | ⚡ | Most powerful |

## 💡 Kullanım Önerileri

### Test/Prototype İçin

```bash
# Hızlı test için MiniROCKET kullanın
python train_model1.py --mode raw --classifier minirocket
python train_model2.py --mode raw --classifier minirocket

# Total time: ~1 saat
# Accuracy: Model 1: ~95-97%, Model 2: ~84-87%
```

### Production İçin

```bash
# En iyi doğruluk için Arsenal kullanın
python train_model1.py --mode raw --classifier arsenal
python train_model2.py --mode raw --classifier arsenal

# Total time: ~4 saat
# Accuracy: Model 1: ~96-98%, Model 2: ~86-90%
```

### Tüm Classifiers'ı Karşılaştırmak İçin

```bash
# Hepsini eğitin ve en iyisini seçin
python train_model1.py --mode raw --classifier all
python train_model2.py --mode raw --classifier all

# Total time: ~15 saat (tüm modeller)
# 6 farklı classifier'ın sonuçlarını karşılaştırabilirsiniz
```

## 📚 Daha Fazla Bilgi

- **Model 1 Detayları**: `model1_binary/README.md`
- **Model 2 Detayları**: `model2_nonstationary/README.md`
- **TRUBA Guide**: `TRUBA_TRAINING_GUIDE.md`

## 🔗 Workflow

```bash
# 1. Veri generation (01-data-generation/)
python generate.py                    # Lokal
sbatch slurm_generate_150k.sh        # TRUBA

# 2. Feature extraction (opsiyonel, 02-feature-engineering/)
python extract_features.py

# 3. Model training (03-models/hierarchical/)
bash submit_training.sh              # TRUBA (önerilen)
# veya
python train_model1.py --mode raw    # Lokal
python train_model2.py --mode raw

# 4. Testing (03-models/hierarchical/)
python test_model1.py --model-path models/model1_arsenal_raw.pkl
python test_model2.py --model-path models/model2_arsenal_raw.pkl
```

## ⚠️ Önemli Notlar

1. **Veri seti gerekli**: Training yapmadan önce `01-data-generation/` ile veri oluşturun
2. **HIVECOTEV2 çok yavaş**: Bu classifier sadece research/benchmark için kullanın
3. **Memory kullanımı**: Model 2, 5-class olduğu için daha fazla RAM kullanır
4. **Parallel training**: Model 1 ve Model 2'yi paralel çalıştırabilirsiniz (bağımsızlar)

## 🆘 Sorun Giderme

### "Dataset not found" hatası

```bash
# Veri setini oluşturun
cd ../../01-data-generation
python generate.py  # veya sbatch slurm_generate_150k.sh
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
