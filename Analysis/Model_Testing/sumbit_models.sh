#!/bin/bash

MODEL_DIR="/home/traaffneu/dansch/Documents/digital_rodent/digitalrodent/Analysis/Model_Testing"

for qmd in "$MODEL_DIR"/*.qmd
do
    name=$(basename "$qmd" .qmd)

    echo "Submitting: $name"

    sbatch \
        --job-name="$name" \
        --nodes=1 \
        --time=12:00:00 \
        --partition=batch \
        --mem=24GB \
        --tmp=75G \
        --output="$MODEL_DIR/${name}_%j.out" \
        --error="$MODEL_DIR/${name}_%j.err" \
        --wrap="
            source /opt/anaconda3/2024.06/etc/profile.d/conda.sh
            conda activate digitalrodent

            cd $MODEL_DIR

            quarto render $qmd
        "
done