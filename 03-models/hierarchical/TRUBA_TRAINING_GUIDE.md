# TRUBA Model Training Guide

Bu dokümanda model eğitimini TRUBA cluster'da nasıl çalıştıracağınız anlatılmaktadır.

## 📋 Ön Gereksinimler

1. **Veri seti oluşturulmuş olmalı**:
   ```bash
   cd /arf/home/iguzel/ts-stationary/hierarchical-ts-classification/01-data-generation
   sbatch slurm_generate_150k.sh  # veya slurm_generate_test.sh
   ```

2. **Conda environment aktif olmalı**:
   ```bash
   module load apps/truba-ai/gpu-2024.0
   conda activate ts-generation
   ```

3. **Gerekli kütüphaneler yüklü olmalı**:
   - scikit-learn >= 1.3.0
   - sktime >= 0.24.0
   - pandas, numpy, pickle

## 🚀 Hızlı Başlangıç

### İnteraktif Mod (Önerilen)

En basit yöntem - interaktif helper script kullanın:

```bash
cd /arf/home/iguzel/ts-stationary/hierarchical-ts-classification/03-models/hierarchical
bash submit_training.sh
```

Bu script size şunları sorar:
1. Hangi modeli eğitmek istiyorsunuz? (Model 1, Model 2, veya her ikisi)
2. Hangi classifier'ı kullanmak istiyorsunuz? (MiniROCKET ⭐, Arsenal ⭐, vs.)

Script otomatik olarak job'ları submit eder ve takip komutlarını gösterir.

### Manuel Mod

Direkt SLURM script'lerini çalıştırmak isterseniz:

```bash
# Model 1 (Binary Classification)
cd model1_binary
sbatch slurm_train_model1.sh

# Model 2 (5-Class Classification)
cd model2_nonstationary
sbatch slurm_train_model2.sh

# Her ikisini sırayla (Model 2, Model 1'in bitmesini bekler)
cd model1_binary
JOB1=$(sbatch slurm_train_model1.sh | awk '{print $4}')
cd ../model2_nonstationary
sbatch --dependency=afterok:$JOB1 slurm_train_model2.sh
```

## ⚙️ SLURM Yapılandırması

### Model 1 (Binary Classification)

**Dosya**: `model1_binary/slurm_train_model1.sh`

```bash
#SBATCH --job-name=train_model1
#SBATCH --partition=orfoz
#SBATCH -c 40              # 40 CPU
#SBATCH --time=06:00:00    # 6 saat limit
```

**Kaynak Kullanımı**:
- CPUs: 40 core (ORFOZ partition)
- Memory: ~20-30 GB (otomatik tahsis)
- Time: 1-5 saat (classifier'a göre)
  - MiniROCKET: ~20-30 dakika
  - ROCKET: ~1 saat
  - Arsenal: ~1.5 saat
  - HIVECOTEV2: ~4-5 saat

### Model 2 (5-Class Classification)

**Dosya**: `model2_nonstationary/slurm_train_model2.sh`

```bash
#SBATCH --job-name=train_model2
#SBATCH --partition=orfoz
#SBATCH -c 40              # 40 CPU
#SBATCH --time=12:00:00    # 12 saat limit
```

**Kaynak Kullanımı**:
- CPUs: 40 core (ORFOZ partition)
- Memory: ~30-40 GB (5-class problem daha fazla RAM kullanır)
- Time: 2-10 saat (classifier'a göre)
  - MiniROCKET: ~30-60 dakika
  - ROCKET: ~2 saat
  - Arsenal: ~2.5 saat
  - HIVECOTEV2: ~8-10 saat

## 🎯 Classifier Seçimi

SLURM script içinde `CLASSIFIER` değişkenini düzenleyerek seçim yapabilirsiniz:

```bash
# Tüm classifier'ları eğit (varsayılan)
CLASSIFIER="all"

# Sadece belirli bir classifier
CLASSIFIER="minirocket"   # ⭐ Hız için önerilen
CLASSIFIER="arsenal"      # ⭐ Doğruluk için önerilen
CLASSIFIER="rocket"       # İyi denge
CLASSIFIER="tsf"          # Baseline
CLASSIFIER="shapelet"     # Yorumlanabilir
CLASSIFIER="hivecote"     # En güçlü (ama çok yavaş)
```

### Öneriler

**Hızlı sonuç için** (test/prototype):
```bash
CLASSIFIER="minirocket"
# Model 1: ~20-30 dakika
# Model 2: ~30-60 dakika
```

**En iyi doğruluk için** (production):
```bash
CLASSIFIER="arsenal"
# Model 1: ~1.5 saat
# Model 2: ~2.5 saat
```

**Hepsini karşılaştırmak için**:
```bash
CLASSIFIER="all"
# Model 1: ~5-6 saat total
# Model 2: ~8-10 saat total
```

## 📊 İzleme ve Kontrol

### Job Durumunu Kontrol Etme

```bash
# Tüm job'larınızı görün
squeue -u iguzel

# Detaylı bilgi
scontrol show job <job_id>
```

### Canlı Log İzleme

```bash
# Model 1 output
tail -f model1_binary/logs/train_model1_<job_id>.out

# Model 2 output
tail -f model2_nonstationary/logs/train_model2_<job_id>.out

# Error logs
tail -f model1_binary/logs/train_model1_<job_id>.err
```

### Job'u İptal Etme

```bash
scancel <job_id>

# Tüm job'larınızı iptal etme
scancel -u iguzel
```

## 📁 Çıktılar

Training tamamlandığında aşağıdaki dosyalar oluşur:

```
model1_binary/
├── models/
│   ├── model1_minirocket_raw.pkl
│   ├── model1_arsenal_raw.pkl
│   └── ...
├── results/
│   ├── model1_minirocket_raw_results.txt
│   └── ...
└── logs/
    ├── train_model1_<job_id>.out
    └── train_model1_<job_id>.err

model2_nonstationary/
├── models/
│   ├── model2_minirocket_raw.pkl
│   └── ...
├── results/
│   ├── model2_minirocket_raw_results.txt
│   └── ...
└── logs/
    ├── train_model2_<job_id>.out
    └── train_model2_<job_id>.err
```

## ⚠️ Önemli Notlar

1. **HIVECOTEV2 çok yavaş**: Bu classifier'ı kullanmadan önce düşünün. Model 1 için ~5 saat, Model 2 için ~10 saat sürebilir.

2. **Memory kullanımı**: Model 2, 5-class problem olduğu için Model 1'den daha fazla RAM kullanır. Eğer out-of-memory hatası alırsanız, daha küçük bir veri seti kullanın veya test-size'ı artırın.

3. **Veri seti boyutu**: 
   - unified-test: ~5K örnekleri (~2-5 dakika training)
   - unified-150k: ~150K örnekleri (~1-10 saat training)

4. **Sequential training**: Model 2'yi çalıştırmadan önce Model 1'i çalıştırmanıza gerek yok. İkisi bağımsız modellerdir.

## 🔧 Sorun Giderme

### Problem: "Dataset not found" hatası
```bash
# Çözüm: Veri setini oluşturun
cd /arf/home/iguzel/ts-stationary/hierarchical-ts-classification/01-data-generation
sbatch slurm_generate_150k.sh
```

### Problem: Job hemen başlamıyor
```bash
# ORFOZ partition'da sıra bekliyor olabilir
squeue -p orfoz  # Partition durumunu kontrol edin
```

### Problem: Out of memory hatası
```bash
# Çözüm 1: Test-size'ı artırın (daha az training data)
# slurm script içinde:
TEST_SIZE=0.3  # veya 0.4

# Çözüm 2: Daha küçük veri seti kullanın
DATA_PATH="../../../data/raw/unified-test"
```

### Problem: Import error (sktime classifier bulunamıyor)
```bash
# Çözüm: sktime'ı güncelleyin
conda activate ts-generation
pip install --upgrade sktime

# Veya sadece o classifier'ı atlayın
CLASSIFIER="minirocket"  # Problemsiz çalışan bir classifier seçin
```

## 📈 Beklenen Performans

### Model 1 (Binary Classification)
- **Accuracy**: 93-98%
- **Training Time** (150K dataset):
  - TimeSeriesForest: ~40 dakika
  - ROCKET: ~1 saat
  - MiniROCKET: ~25 dakika ⭐
  - Arsenal: ~1.5 saat
  - Shapelet: ~2 saat
  - HIVECOTEV2: ~5 saat

### Model 2 (5-Class Classification)
- **Accuracy**: 80-92%
- **Training Time** (150K non-stationary samples):
  - TimeSeriesForest: ~1 saat
  - ROCKET: ~2 saat
  - MiniROCKET: ~45 dakika ⭐
  - Arsenal: ~2.5 saat
  - Shapelet: ~3 saat
  - HIVECOTEV2: ~10 saat

## 🎓 Workflow Önerisi

**İlk kez çalıştırıyorsanız**:
```bash
# 1. Küçük test veri seti ile başlayın
cd 01-data-generation
sbatch slurm_generate_test.sh  # ~1 saat

# 2. MiniROCKET ile hızlı test
cd ../03-models/hierarchical
bash submit_training.sh
# Seçimler: Model 1 → MiniROCKET

# 3. Sonuçlar iyi görünüyorsa, büyük veri seti
cd ../../01-data-generation
sbatch slurm_generate_150k.sh  # ~12-18 saat

# 4. Production training
cd ../03-models/hierarchical
bash submit_training.sh
# Seçimler: Both Models → Arsenal (en iyi doğruluk)
```

**Production için**:
```bash
# Arsenal ile her iki modeli de eğitin
CLASSIFIER="arsenal"
# Model 1: ~1.5 saat, %96-98 accuracy
# Model 2: ~2.5 saat, %86-90 accuracy
# Total: ~4 saat, hierarchical system hazır
```

---

**Hazırlayan**: GitHub Copilot  
**Son Güncelleme**: 31 Ekim 2025  
**TRUBA Partition**: ORFOZ (110 CPUs)
