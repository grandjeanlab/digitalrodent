#!/bin/bash
#SBATCH --job-name=hbr_digitalrodent
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=hbr_%j.log

cd ~/digitalrodent/Analysis
source .venv/bin/activate

# No Python headers on the nodes: run PyTensor without C compilation
export PYTENSOR_FLAGS="cxx="

export HBR_ALL=0          # 0 = only the selected components, 1 = all 18
export HBR_MODELS=A,B
export HBR_CHAINS=4
export HBR_DRAWS=1500
export HBR_TUNE=500

quarto render Model_Training_HBR.qmd
