# 🚀 Model 1 Quick Start Guide

## Model 1: Binary Classification (Stationary vs Unstationary)

---

## ⚡ Hızlı Başlangıç

### 1. Gerekli Kütüphaneleri Yükle

```bash
pip install sktime scikit-learn numpy pandas
```

### 2. Veri Hazırla (Eğer Yoksa)

```bash
# Test verisi oluştur
cd 01-data-generation
python3 generate_test.py
```

> Not: Etiket oluşturma artık OPSIYONEL. Yeni ts-stationary çıktıları `is_stationary`, `primary_category`, `sub_category` içerir; eğitim scripti bu alanlardan etiketleri otomatik türetir. İstersen yine de tek bir labeled parquet üretmek için 02-preprocessing/create_labels.py kullanabilirsin.

### 3. Model 1'i Eğit

```bash
cd 03-models/hierarchical/model1_binary
python3 train_model1.py
```

**Beklenen Süre:** 5-15 dakika  
**Beklenen Accuracy:** >95%

### 4. (Opsiyonel) Test Et

```bash
python3 test_model1.py
```

---

## 📊 Ne Yapar?

### Input:
- Time series data (her series 1000-5000 nokta)
- Total: ~15,000 series
- Split: 80% train (12,000), 20% test (3,000)

### Process:
1. **Veriyi yükler**: İşlenmiş labeled dosya varsa oradan; yoksa raw parquet’lerden
2. **Etiketleri türetir**: `is_stationary`, `primary_category`, `sub_category` → Model1/2/3
3. **Hazırlar**: 1,500 noktalık fixed length'e çevirir
4. **Eğitir**: TimeSeriesForest ve ROCKET
5. **Karşılaştırır**: En iyi modeli seçer ve kaydeder

### Output:
- **Stationary (0)**: AR, MA, ARMA, White Noise
- **Unstationary (1)**: Trend, Volatility, Stochastic, Anomaly, Break

---

## 📁 Çıktı Dosyaları

```
03-models/hierarchical/model1_binary/
└── saved_models/
    ├── model1_binary_classifier.pkl    # Eğitilmiş model
    └── model1_metadata.pkl              # Model bilgileri
```

---

## 🐛 Sorun Giderme

### "Data file not found"
Veri üret: `cd 01-data-generation && python3 generate_test.py`

### "sktime not installed"
`pip install sktime`

### Matplotlib backend hataları (macOS)
GUI’siz backend kullan: `matplotlib.use("Agg")` veya çıktıyı dosyaya kaydet.

---

## ⏭️ Sonraki Adımlar

Model 1 başarılı olduktan sonra:
1. ✅ Model 1 accuracy >95% ise devam
2. ⏭️ Model 2 (5-class unstationary types)
3. ⏭️ Model 3 (subtypes)
4. ⏭️ Cascade pipeline

---

**Hazır mısın? Hadi başlayalım!** 🚀

```bash
cd 03-models/hierarchical/model1_binary
python3 train_model1.py
```

