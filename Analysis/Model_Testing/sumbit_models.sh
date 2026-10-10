#!/bin/bash
#SBATCH --job-name=model_testing
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --mem=24GB
#SBATCH --tmp=75G
#SBATCH --output=model_testing_%j.out

source /opt/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate digitalrodent

cd /home/traaffneu/dansch/Documents/digital_rodent/digitalrodent/Analysis/Model_Testing

echo "Running 01_Age_only"
quarto render 01_Age_only.qmd

echo "Running 02_Sex_only"
quarto render 02_Sex_only.qmd

echo "Running 03_Age_Sex"
quarto render 03_Age_Sex.qmd

echo "Running 04_Age_Sex_comp2_10_11"
quarto render 04_Age_Sex_comp2_10_11.qmd

echo "Running 05_Age_Sex_Strain"
quarto render 05_Age_Sex_Strain.qmd

echo "Running 06_Age_Sex_MRI_Coil"
quarto render 06_Age_Sex_MRI_Coil.qmd

echo "Running 07_Age_Sex_Strain_MRI_Coil"
quarto render 07_Age_Sex_Strain_MRI_Coil.qmd

echo "Running 08_Age_Sex_Acquisition"
quarto render 08_Age_Sex_Acquisition.qmd

echo "Running 09_Full_NoBatch"
quarto render 09_Full_NoBatch.qmd

echo "ALL MODELS FINISHED"
