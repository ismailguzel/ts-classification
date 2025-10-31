#!/bin/bash
# TRUBA-specific job submission helper

echo "=========================================="
echo "TRUBA Time Series Dataset Generation"
echo "=========================================="
echo ""
echo "Current location: $(pwd)"
echo "User: $USER"
echo ""
echo "Available options:"
echo "  1) Generate test dataset (~1.5K, ~10 min, 40 CPUs)"
echo "  2) Generate full dataset (150K, ~15 hrs, 110 CPUs)"
echo "  3) Generate both (test first, then full - dependency)"
echo "  4) Check job status"
echo "  5) Check detailed job info"
echo "  6) Cancel all jobs"
echo "  7) View latest logs"
echo ""
read -p "Select option [1-7]: " option

case $option in
    1)
        echo "Submitting test dataset job to ORFOZ partition..."
        sbatch slurm_generate_test.sh
        echo ""
        echo "Job submitted! Check status with:"
        echo "  squeue -u $USER"
        echo "  tail -f logs/generate_test_*.out"
        ;;
    2)
        echo "Submitting full 150K dataset job to ORFOZ partition..."
        echo ""
        echo "⚠️  WARNING:"
        echo "  - Duration: ~15 hours"
        echo "  - CPUs: 110 (full ORFOZ node)"
        echo "  - Storage: ~10 GB"
        echo ""
        read -p "Continue? [y/N]: " confirm
        if [[ $confirm == [yY] ]]; then
            sbatch slurm_generate_150k.sh
            echo ""
            echo "Job submitted! Monitor with:"
            echo "  watch -n 60 squeue -u $USER"
            echo "  tail -f logs/generate_150k_*.out"
        else
            echo "Cancelled."
        fi
        ;;
    3)
        echo "Submitting test job first..."
        TEST_JOB=$(sbatch --parsable slurm_generate_test.sh)
        echo "✓ Test job submitted: $TEST_JOB"
        echo ""
        echo "Submitting full job (depends on test completion)..."
        FULL_JOB=$(sbatch --parsable --dependency=afterok:$TEST_JOB slurm_generate_150k.sh)
        echo "✓ Full job submitted: $FULL_JOB"
        echo ""
        echo "Jobs chained! Full job will start after test completes successfully."
        echo ""
        echo "Monitor with:"
        echo "  squeue -u $USER"
        ;;
    4)
        echo "Checking job status..."
        echo ""
        squeue -u $USER
        echo ""
        echo "Legend:"
        echo "  PD = Pending (waiting)"
        echo "  R  = Running"
        echo "  CG = Completing"
        ;;
    5)
        echo "Enter job ID:"
        read jobid
        scontrol show job $jobid
        ;;
    6)
        echo "⚠️  This will cancel ALL your jobs!"
        read -p "Are you sure? [y/N]: " confirm
        if [[ $confirm == [yY] ]]; then
            scancel -u $USER
            echo "All jobs cancelled."
        else
            echo "Cancelled."
        fi
        ;;
    7)
        echo "Available log files:"
        echo ""
        ls -lht logs/*.out 2>/dev/null | head -5
        echo ""
        echo "View which log?"
        echo "  1) Latest test log"
        echo "  2) Latest 150K log"
        read -p "Select [1-2]: " logopt
        case $logopt in
            1)
                latest=$(ls -t logs/generate_test_*.out 2>/dev/null | head -1)
                if [ -n "$latest" ]; then
                    echo "Viewing: $latest"
                    tail -100 "$latest"
                else
                    echo "No test logs found."
                fi
                ;;
            2)
                latest=$(ls -t logs/generate_150k_*.out 2>/dev/null | head -1)
                if [ -n "$latest" ]; then
                    echo "Viewing: $latest"
                    tail -100 "$latest"
                else
                    echo "No 150K logs found."
                fi
                ;;
        esac
        ;;
    *)
        echo "Invalid option."
        ;;
esac
