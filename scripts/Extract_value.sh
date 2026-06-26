#!/bin/bash
module load afni/23.3.02

# Parameters
ICA_map="/project/4180000.73/template/mouse/ica.nii.gz"
DR_base="/project/4180000.73/University_of_Florida_Mouse_Sepsis_fMRI_StudyT/output_mouse/commonspace_analysis_datasink/dual_regression_nii/"
output_beta="/project/4180000.73/University_of_Florida_Mouse_Sepsis_fMRI_StudyT/output_mouse/DR_results_beta.csv"
output_zscore="/project/4180000.73/University_of_Florida_Mouse_Sepsis_fMRI_StudyT/output_mouse/DR_results_zscore.csv"
tmp_dir="/project/4180000.73/University_of_Florida_Mouse_Sepsis_fMRI_StudyT/output_mouse/tmp_DR_$$"
n_comp=18

# Create temp directory
mkdir -p ${tmp_dir}

# Step 1 - Resample ICA once
echo "Step 1: Resampling ICA..."
first_DR=$(find ${DR_base} -name "*DR_maps.nii.gz" | head -1)
echo "Using reference: ${first_DR}"
3dresample -input ${ICA_map} \
           -master ${first_DR} \
           -prefix ${tmp_dir}/ica_res.nii.gz

# Check Step 1 worked
if [ ! -f ${tmp_dir}/ica_res.nii.gz ]; then
    echo "ERROR: 3dresample failed!"
    exit 1
fi
echo "Step 1 OK"

# Step 2 - Split resampled ICA into components
echo "Step 2: Splitting ICA..."
fslsplit ${tmp_dir}/ica_res.nii.gz ${tmp_dir}/ica_comp_ -t

# Check Step 2 worked
if [ ! -f ${tmp_dir}/ica_comp_0000.nii.gz ]; then
    echo "ERROR: fslsplit ICA failed!"
    exit 1
fi
echo "Step 2 OK"

# Step 3 - Create masks once for all components
echo "Step 3: Creating masks..."
for i in $(seq 0 $((n_comp-1)))
do
    idx=$(printf "%04d" $i)
    fslmaths ${tmp_dir}/ica_comp_${idx}.nii.gz -thrp 90 -bin ${tmp_dir}/mask_comp_${idx}.nii.gz
    if [ ! -f ${tmp_dir}/mask_comp_${idx}.nii.gz ]; then
        echo "ERROR: mask creation failed for component ${idx}!"
        exit 1
    fi
done
echo "Step 3 OK - all masks created"

# Step 4 - Write CSV headers
header="subject"
for i in $(seq 1 ${n_comp}); do header="${header},comp${i}"; done
echo ${header} > ${output_beta}
echo ${header} > ${output_zscore}

# Step 5 - Loop over all subjects
find ${DR_base} -name "*DR_maps.nii.gz" | sort | while read DR_map
do
    sub=$(basename ${DR_map} | cut -d'_' -f1-2)
    echo "Processing ${sub}..."

    fslsplit ${DR_map} ${tmp_dir}/DR_comp_ -t

    row_beta="${sub}"
    row_zscore="${sub}"

    for i in $(seq 0 $((n_comp-1)))
    do
        idx=$(printf "%04d" $i)

        mean_beta=$(fslmeants -i ${tmp_dir}/DR_comp_${idx}.nii.gz \
                              -m ${tmp_dir}/mask_comp_${idx}.nii.gz)

        dr_mean=$(fslstats ${tmp_dir}/DR_comp_${idx}.nii.gz -M)
        dr_std=$(fslstats ${tmp_dir}/DR_comp_${idx}.nii.gz -S)

        fslmaths ${tmp_dir}/DR_comp_${idx}.nii.gz \
                 -sub ${dr_mean} \
                 -div ${dr_std} \
                 ${tmp_dir}/DR_comp_${idx}_Z.nii.gz

        mean_zscore=$(fslmeants -i ${tmp_dir}/DR_comp_${idx}_Z.nii.gz \
                                -m ${tmp_dir}/mask_comp_${idx}.nii.gz)

        row_beta="${row_beta},${mean_beta}"
        row_zscore="${row_zscore},${mean_zscore}"
    done

    echo ${row_beta} >> ${output_beta}
    echo ${row_zscore} >> ${output_zscore}

    rm -f ${tmp_dir}/DR_comp_*.nii.gz
    echo "${sub} done"
done

# Final cleanup
rm -rf ${tmp_dir}

echo "Done!"
echo "Beta results:    ${output_beta}"
echo "Z-score results: ${output_zscore}"
