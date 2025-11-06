#!/bin/bash
#SBATCH --job-name=ts_gen
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err
#SBATCH --partition=orfoz
#SBATCH --reservation=test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 55
#SBATCH -C weka
#SBATCH --time=20:00:00
#SBATCH --account=iguzel

# Accept scale as first argument (default: 90k)
SCALE="${1:-90k}"

# Job bilgisi
echo "=================================================="
echo "TRUBA Time Series Generation - $SCALE Dataset"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Scale: $SCALE"
echo "Start Time: $(date)"
echo "=================================================="

# Çalışma dizinine git
cd /arf/home/iguzel/ts-stationary/hierarchical-ts-classification/01-data-generation
echo "Working directory: $(pwd)"

# Logs klasörünü oluştur
mkdir -p logs

# TRUBA modüllerini yükle
echo ""
echo "Loading TRUBA modules..."
module purge
module load apps/truba-ai/gpu-2024.0
module list

# Conda environment aktive et
echo ""
echo "Activating conda environment..."
conda activate ts-generation
echo "Conda env: $CONDA_DEFAULT_ENV"

# Python ve kütüphane versiyonlarını kontrol et
echo ""
echo "Environment Info:"
python --version
python -c "import sys; print(f'Python path: {sys.executable}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import timeseries_dataset_generator; print('ts-stationary: OK')"

# Config'i göster (specific scale)
echo ""
echo "Configuration for scale: $SCALE"
python config.py "$SCALE"

# Data generation başlat
echo ""
echo "=================================================="
echo "Starting $SCALE dataset generation..."
echo "Using 55 CPUs on ORFOZ partition"
echo "=================================================="

# Full path kullan
SCRIPT_PATH="/arf/home/iguzel/ts-stationary/hierarchical-ts-classification/01-data-generation/generate.py"
echo "Script path: $SCRIPT_PATH"
echo "Scale: $SCALE"
echo ""

# Time ve resource tracking ile çalıştır
time python -u "$SCRIPT_PATH" --scale "$SCALE"

# Sonuçları kontrol et
echo ""
echo "=================================================="
echo "Generation Complete!"
echo "End Time: $(date)"
echo "=================================================="

# Output klasörünü kontrol et
OUTPUT_DIR="/arf/home/iguzel/ts-stationary/hierarchical-ts-classification/data/raw/unified-${SCALE}"
if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "Output Directory Statistics:"
    echo "  Location: $OUTPUT_DIR"
    echo "  File count: $(find $OUTPUT_DIR -type f -name "*.parquet" | wc -l)"
    echo "  Total size: $(du -sh $OUTPUT_DIR 2>/dev/null | cut -f1)"
    echo ""
    echo "Sample files (first 20):"
    find $OUTPUT_DIR -name "*.parquet" 2>/dev/null | head -20
    echo ""
    echo "Category breakdown:"
    for cat in stationary deterministic_trend volatility stochastic point_anomaly collective_anomaly multi_mean_shift multi_variance_shift multi_trend_shift; do
        count=$(find $OUTPUT_DIR -name "*${cat}*.parquet" 2>/dev/null | wc -l)
        if [ $count -gt 0 ]; then
            echo "  $cat: $count files"
        fi
    done
fi

echo ""
echo "Job completed successfully!"
echo "Results saved to: $OUTPUT_DIR"