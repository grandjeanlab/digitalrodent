#!/bin/bash

# author: Giulia Vasirani
# intial date: 21.05.2026

#load AFNI
module load afni/2022

# Parameters
ICA_map="/project/4180000.73/template/mouse/ica.nii.gz"
DR_base="/project/4180000.73/Resting_state_fMRI_Alvino/output_mouse/commonspace_analysis_datasink/dual_regression_nii"
output="/project/4180000.73/Resting_state_fMRI_Alvino/output_mouse/DR_results_all_subjects.csv"
tmp_base="/tmp/DR_extraction_$$"
n_comp=18

mkdir -p ${tmp_base}

# Step 1 & 2 - Setup global masks
first_DR=$(find ${DR_base} -name "*DR_maps.nii.gz" | head -1)
3dresample -input ${ICA_map} -master ${first_DR} -prefix ${tmp_base}/ica_res.nii.gz
fslsplit ${tmp_base}/ica_res.nii.gz ${tmp_base}/mask_base_ -t

for i in $(seq 0 $((n_comp-1))); do
    idx=$(printf "%04d" $i)
    fslmaths ${tmp_base}/mask_base_${idx}.nii.gz -thrp 90 -bin ${tmp_base}/mask_comp_${idx}.nii.gz
done

# Step 4 - Write CSV header
header="subject"
for i in $(seq 1 ${n_comp}); do header="${header},comp${i}"; done
echo ${header} > ${output}

# Step 5 - Loop over subjects
find ${DR_base} -name "*DR_maps.nii.gz" | sort | while read DR_map; do
    sub=$(basename ${DR_map} | cut -d'_' -f1-2)
    sub_tmp="${tmp_base}/${sub}"
    mkdir -p ${sub_tmp}
    
    echo "Processing ${sub}..."
    
    # Split subject DR map into its own folder
    fslsplit ${DR_map} ${sub_tmp}/DR_comp_ -t
    
    # Extract mean per component
    row="${sub}"
    for i in $(seq 0 $((n_comp-1))); do
        idx=$(printf "%04d" $i)
        # Verify the file exists before running fslmeants
        if [ -f "${sub_tmp}/DR_comp_${idx}.nii.gz" ]; then
            mean=$(fslmeants -i ${sub_tmp}/DR_comp_${idx}.nii.gz -m ${tmp_base}/mask_comp_${idx}.nii.gz)
            row="${row},${mean}"
        else
            row="${row},NaN"
        fi
    done
    
    echo ${row} >> ${output}
    rm -rf ${sub_tmp} # Clean up only this subject's files
    echo "${sub} done"
done

rm -rf ${tmp_base}
echo "Done! Results saved in ${output}"
