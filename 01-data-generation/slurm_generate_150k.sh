#!/bin/bash
#SBATCH --job-name=ts_gen_150k
#SBATCH --output=logs/generate_150k_%j.out
#SBATCH --error=logs/generate_150k_%j.err
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=110
#SBATCH --constraint=weka
#SBATCH --time=20:00:00
#SBATCH --account=iguzel

# Job bilgisi
echo "=================================================="
echo "TRUBA Time Series Generation - 150K Dataset"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
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

# Config'i göster
echo ""
echo "Configuration:"
python -c "from config import COUNTS_150K; print(f'Total samples: 150,000'); print(f'Stationary: {COUNTS_150K[\"stationary\"][\"total\"]:,}'); print(f'Non-stationary: 75,000')"

# Data generation başlat
echo ""
echo "=================================================="
echo "Starting 150K dataset generation..."
echo "Expected duration: 12-18 hours"
echo "Expected size: ~7-10 GB"
echo "Using 110 CPUs on ORFOZ partition"
echo "=================================================="

# Time ve resource tracking ile çalıştır
/usr/bin/time -v python generate.py

# Sonuçları kontrol et
echo ""
echo "=================================================="
echo "Generation Complete!"
echo "End Time: $(date)"
echo "=================================================="

# Output klasörünü kontrol et
OUTPUT_DIR="../data/raw/unified-150k"
if [ -d "$OUTPUT_DIR" ]; then
    echo ""
    echo "Output Directory Statistics:"
    echo "  Location: $OUTPUT_DIR"
    echo "  File count: $(find $OUTPUT_DIR -type f -name "*.parquet" 2>/dev/null | wc -l)"
    echo "  Total size: $(du -sh $OUTPUT_DIR 2>/dev/null | cut -f1)"
    echo ""
    echo "Sample files (first 20):"
    find $OUTPUT_DIR -name "*.parquet" 2>/dev/null | head -20
    echo ""
    echo "Category breakdown:"
    for dir in stationary trend volatility stochastic anomaly structural; do
        count=$(find $OUTPUT_DIR -type d -name "*${dir}*" 2>/dev/null | wc -l)
        if [ $count -gt 0 ]; then
            echo "  $dir: $count directories"
        fi
    done
fi

echo ""
echo "Job completed successfully!"
echo "Results saved to: $OUTPUT_DIR"
