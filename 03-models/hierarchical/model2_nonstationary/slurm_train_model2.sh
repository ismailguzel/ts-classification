#!/bin/bash
#SBATCH --job-name=train_model2
#SBATCH --output=logs/train_model2_%j.out
#SBATCH --error=logs/train_model2_%j.err
#SBATCH --partition=orfoz
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 110
#SBATCH -C weka
#SBATCH --time=12:00:00
#SBATCH --account=iguzel

# Job bilgisi
echo "=================================================="
echo "TRUBA Model 2 Training - 5-Class Classification"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start Time: $(date)"
echo "=================================================="

# Çalışma dizinine git
cd /arf/home/iguzel/ts-stationary/hierarchical-ts-classification/03-models/hierarchical/model2_nonstationary
echo "Working directory: $(pwd)"

# Logs klasörünü oluştur
mkdir -p logs
mkdir -p models

# TRUBA modüllerini yükle
echo ""
echo "Loading TRUBA modules..."
module purge
module load apps/truba-ai/gpu-2024.0
module list

# Conda environment aktive et
echo ""
echo "Activating conda environment..."
conda activate ts-sktime
echo "Conda env: $CONDA_DEFAULT_ENV"

# Python ve kütüphane versiyonlarını kontrol et
echo ""
echo "Environment Info:"
python --version
python -c "import sys; print(f'Python path: {sys.executable}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python -c "import sktime; print(f'sktime: {sktime.__version__}')"

# Veri setini kontrol et
DATA_PATH="../../../data/raw/unified-90k"
echo ""
echo "=================================================="
echo "Dataset Info:"
echo "  Location: $DATA_PATH"
if [ -d "$DATA_PATH" ]; then
    echo "  File count: $(find $DATA_PATH -type f -name "*.parquet" | wc -l)"
    echo "  Total size: $(du -sh $DATA_PATH 2>/dev/null | cut -f1)"
    
    # Non-stationary kategorileri say
    echo ""
    echo "  Non-stationary categories:"
    for cat in trend volatility stochastic anomaly structural_break; do
        count=$(find $DATA_PATH -name "*${cat}*.parquet" 2>/dev/null | wc -l)
        echo "    $cat: $count files"
    done
    echo "  ✓ Dataset found"
else
    echo "  ✗ ERROR: Dataset not found!"
    echo "  Please generate data first: cd 01-data-generation && sbatch slurm_generate_150k.sh"
    exit 1
fi

# Model training parametreleri
MODE="raw"
CLASSIFIER="rocket"  # Fast and reliable (minirocket removed due to NumPy 2.0 issues)
TEST_SIZE=0.2

echo ""
echo "=================================================="
echo "Training Configuration:"
echo "  Mode: $MODE"
echo "  Classifier: $CLASSIFIER (memory-efficient, single classifier)"
echo "  Test Size: $TEST_SIZE"
echo "  Data Path: $DATA_PATH"
echo "  Note: Using single classifier to prevent OOM errors"
echo "=================================================="

# Model 2 eğitimi başlat
echo ""
echo "=================================================="
echo "Starting Model 2 Training..."
echo "Task: 5-Class Classification (Trend/Volatility/Stochastic/Anomaly/Structural)"
echo "Expected duration: 2-10 hours (depends on classifier)"
echo "Note: 5-class problem is harder than binary - takes longer"
echo "Using 110 CPUs on ORFOZ partition"
echo "=================================================="

# Time ve resource tracking ile çalıştır
python -u train_model2.py \
    --mode $MODE \
    --classifier $CLASSIFIER \
    --data-path $DATA_PATH \
    --test-size $TEST_SIZE

# Training sonuçlarını kontrol et
TRAINING_EXIT_CODE=$?
echo ""
echo "=================================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "Exit Code: $TRAINING_EXIT_CODE"
echo "=================================================="

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    # Kaydedilen modelleri listele
    echo ""
    echo "Saved Models:"
    if [ -d "models" ]; then
        ls -lh models/model2_*.pkl 2>/dev/null || echo "  No model files found"
    fi
    
    # Results klasörünü kontrol et
    if [ -d "results" ]; then
        echo ""
        echo "Results:"
        ls -lh results/model2_*.txt 2>/dev/null || echo "  No result files found"
    fi
    
    echo ""
    echo "✓ Job completed successfully!"
else
    echo ""
    echo "✗ Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "Check logs for details: logs/train_model2_${SLURM_JOB_ID}.err"
    exit $TRAINING_EXIT_CODE
fi
