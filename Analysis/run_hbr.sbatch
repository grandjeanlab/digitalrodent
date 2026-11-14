#!/bin/bash
#SBATCH --job-name=hbr_digitalrodent
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=hbr_%j.log
# Add a --partition line here if your cluster requires one.

# HBR normative models (Model_Training_HBR.qmd)
# Submit from ~/digitalrodent/Analysis with:   sbatch run_hbr.sbatch

cd ~/digitalrodent/Analysis
source .venv/bin/activate

export HBR_ALL=0          # 0 = only the selected components (faster), 1 = all 18
export HBR_MODELS=A,B     # models to fit
export HBR_CHAINS=4
export HBR_DRAWS=1500
export HBR_TUNE=500

quarto render Model_Training_HBR.qmd
