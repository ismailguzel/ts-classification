#!/bin/bash

# ========================================
# TRUBA Model Training Submission Helper
# ========================================

echo "=================================================="
echo "TRUBA Model Training Job Submission"
echo "=================================================="
echo ""

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Model seçimi
echo -e "${BLUE}Available models:${NC}"
echo "  1) Model 1 - Binary Classification (Stationary vs Non-Stationary)"
echo "     └─ Time: ~1-5 hours | CPUs: 40 | Accuracy: 93-98%"
echo ""
echo "  2) Model 2 - 5-Class Classification (Non-Stationary Types)"
echo "     └─ Time: ~2-10 hours | CPUs: 40 | Accuracy: 80-92%"
echo ""
echo "  3) Both Models (Sequential Training)"
echo "     └─ Time: ~3-15 hours total"
echo ""
echo "  4) Exit"
echo ""

read -p "Select model to train [1-4]: " MODEL_CHOICE

case $MODEL_CHOICE in
    1)
        echo ""
        echo -e "${GREEN}Selected: Model 1 (Binary Classification)${NC}"
        
        # Classifier seçimi
        echo ""
        echo -e "${BLUE}Available classifiers:${NC}"
        echo "  1) All classifiers (trains all 4 models)"
        echo "  2) TimeSeriesForest (fast, baseline)"
        echo "  3) ROCKET ⭐ (good balance, recommended)"
        echo "  4) Arsenal ⭐ (best accuracy)"
        echo "  5) Shapelet (interpretable)"
        echo ""
        
        read -p "Select classifier [1-5]: " CLASSIFIER_CHOICE
        
        case $CLASSIFIER_CHOICE in
            1) CLASSIFIER="all" ;;
            2) CLASSIFIER="tsf" ;;
            3) CLASSIFIER="rocket" ;;
            4) CLASSIFIER="arsenal" ;;
            5) CLASSIFIER="shapelet" ;;
            *) 
                echo -e "${RED}Invalid choice. Exiting.${NC}"
                exit 1
                ;;
        esac
        
        echo ""
        echo -e "${YELLOW}Submitting Model 1 training job...${NC}"
        echo "  Classifier: $CLASSIFIER"
        
        cd model1_binary
        mkdir -p logs models results
        
        # SLURM script'i güncelle
        sed -i.bak "s/CLASSIFIER=\".*\"/CLASSIFIER=\"$CLASSIFIER\"/" slurm_train_model1.sh
        
        # Job submit
        JOB_ID=$(sbatch slurm_train_model1.sh | awk '{print $4}')
        
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}✓ Job submitted successfully!${NC}"
            echo "  Job ID: $JOB_ID"
            echo ""
            echo "Monitor job with:"
            echo "  squeue -u iguzel"
            echo "  tail -f logs/train_model1_${JOB_ID}.out"
        else
            echo -e "${RED}✗ Job submission failed${NC}"
            exit 1
        fi
        ;;
        
    2)
        echo ""
        echo -e "${GREEN}Selected: Model 2 (5-Class Classification)${NC}"
        
        # Classifier seçimi
        echo ""
        echo -e "${BLUE}Available classifiers:${NC}"
        echo "  1) All classifiers (trains all 4 models)"
        echo "  2) TimeSeriesForest (fast, baseline)"
        echo "  3) ROCKET ⭐ (good balance, recommended)"
        echo "  4) Arsenal ⭐ (best accuracy)"
        echo "  5) Shapelet (interpretable)"
        echo ""
        
        read -p "Select classifier [1-5]: " CLASSIFIER_CHOICE
        
        case $CLASSIFIER_CHOICE in
            1) CLASSIFIER="all" ;;
            2) CLASSIFIER="tsf" ;;
            3) CLASSIFIER="rocket" ;;
            4) CLASSIFIER="arsenal" ;;
            5) CLASSIFIER="shapelet" ;;
            *) 
                echo -e "${RED}Invalid choice. Exiting.${NC}"
                exit 1
                ;;
        esac
        
        echo ""
        echo -e "${YELLOW}Submitting Model 2 training job...${NC}"
        echo "  Classifier: $CLASSIFIER"
        
        cd model2_nonstationary
        mkdir -p logs models results
        
        # SLURM script'i güncelle
        sed -i.bak "s/CLASSIFIER=\".*\"/CLASSIFIER=\"$CLASSIFIER\"/" slurm_train_model2.sh
        
        # Job submit
        JOB_ID=$(sbatch slurm_train_model2.sh | awk '{print $4}')
        
        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}✓ Job submitted successfully!${NC}"
            echo "  Job ID: $JOB_ID"
            echo ""
            echo "Monitor job with:"
            echo "  squeue -u iguzel"
            echo "  tail -f logs/train_model2_${JOB_ID}.out"
        else
            echo -e "${RED}✗ Job submission failed${NC}"
            exit 1
        fi
        ;;
        
    3)
        echo ""
        echo -e "${GREEN}Selected: Both Models (Sequential)${NC}"
        
        # Classifier seçimi
        echo ""
        echo -e "${BLUE}Select classifier for both models:${NC}"
        echo "  1) ROCKET ⭐ (recommended for balance)"
        echo "  2) Arsenal ⭐ (recommended for accuracy)"
        echo "  3) TimeSeriesForest (fast baseline)"
        echo "  4) Shapelet (interpretable)"
        echo ""
        
        read -p "Select classifier [1-4]: " CLASSIFIER_CHOICE
        
        case $CLASSIFIER_CHOICE in
            1) CLASSIFIER="rocket" ;;
            2) CLASSIFIER="arsenal" ;;
            3) CLASSIFIER="tsf" ;;
            4) CLASSIFIER="shapelet" ;;
            *) 
                echo -e "${RED}Invalid choice. Exiting.${NC}"
                exit 1
                ;;
        esac
        
        echo ""
        echo -e "${YELLOW}Submitting both training jobs...${NC}"
        echo "  Classifier: $CLASSIFIER"
        
        # Model 1
        cd model1_binary
        mkdir -p logs models results
        sed -i.bak "s/CLASSIFIER=\".*\"/CLASSIFIER=\"$CLASSIFIER\"/" slurm_train_model1.sh
        JOB1_ID=$(sbatch slurm_train_model1.sh | awk '{print $4}')
        cd ..
        
        # Model 2 (depends on Model 1)
        cd model2_nonstationary
        mkdir -p logs models results
        sed -i.bak "s/CLASSIFIER=\".*\"/CLASSIFIER=\"$CLASSIFIER\"/" slurm_train_model2.sh
        JOB2_ID=$(sbatch --dependency=afterok:$JOB1_ID slurm_train_model2.sh | awk '{print $4}')
        cd ..
        
        echo ""
        echo -e "${GREEN}✓ Jobs submitted successfully!${NC}"
        echo "  Model 1 Job ID: $JOB1_ID"
        echo "  Model 2 Job ID: $JOB2_ID (will start after Model 1)"
        echo ""
        echo "Monitor jobs with:"
        echo "  squeue -u iguzel"
        ;;
        
    4)
        echo "Exiting..."
        exit 0
        ;;
        
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "Useful Commands:"
echo "=================================================="
echo "Check job status:"
echo "  squeue -u iguzel"
echo ""
echo "Cancel a job:"
echo "  scancel <job_id>"
echo ""
echo "View live output:"
echo "  tail -f model1_binary/logs/train_model1_<job_id>.out"
echo "  tail -f model2_nonstationary/logs/train_model2_<job_id>.out"
echo ""
echo "Check results after completion:"
echo "  ls -lh model1_binary/models/"
echo "  ls -lh model2_nonstationary/models/"
echo "=================================================="
