#!/bin/bash
module load afni/23.3.02

#Set the folders
ICA_map="/project/4180000.73/template/mouse/ica.nii.gz"
DR_base="/project/4180000.73/Mouse_rest_anesthesia/output_mouse/commonspace_analysis_datasink/dual_regression_nii/"
BRAIN_MASK="/project/4180000.73/template/mouse/mask.nii.gz"    
output="/project/4180000.73/Mouse_rest_anesthesia/output_mouse//DR_results.csv"
tmp_dir="/project/4180000.73/Mouse_rest_anesthesia/output_mouse/tmp_DR_$$"
n_comp=18

mkdir -p ${tmp_dir}

# Step 1 - Resample ICA 
echo "Step 1: Resampling ICA..."
first_DR=$(find ${DR_base} -name "*DR_maps.nii.gz" | sort | head -1)
echo "Using reference: ${first_DR}"
3dresample -input ${ICA_map} \
           -master ${first_DR} \
           -prefix ${tmp_dir}/ica_res.nii.gz
if [ ! -f ${tmp_dir}/ica_res.nii.gz ]; then
    echo "ERROR: 3dresample failed!"
    exit 1
fi
echo "Step 1 OK"

# Step 2 - Split ICA 
echo "Step 2: Splitting ICA..."
fslsplit ${tmp_dir}/ica_res.nii.gz ${tmp_dir}/ica_comp_ -t
if [ ! -f ${tmp_dir}/ica_comp_0000.nii.gz ]; then
    echo "ERROR: fslsplit ICA failed!"
    exit 1
fi
echo "Step 2 OK"

# Step 3 - Create masks
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

# Step 4 - Header CSV
header="subject"
for i in $(seq 1 ${n_comp}); do header="${header},comp${i}"; done
echo ${header} > ${output}

# Step 5 - Loop over subjects
echo "Step 5: Processing subjects..."
find ${DR_base} -name "*DR_maps.nii.gz" | sort | while read DR_map
do
    sub=$(basename ${DR_map} | cut -d'_' -f1-2)
    echo "Processing ${sub}..."
    
    # Split subject's 4D DR map into 3D volumes
    fslsplit ${DR_map} ${tmp_dir}/DR_comp_ -t
    
    row="${sub}"
    for i in $(seq 0 $((n_comp-1)))
    do
        idx=$(printf "%04d" $i)
        mean=$(fslmeants -i ${tmp_dir}/DR_comp_${idx}.nii.gz -m ${tmp_dir}/mask_comp_${idx}.nii.gz)
        row="${row},${mean}"
    done
    
    # Append row to CSV
    echo ${row} >> ${output}
    
    # Clean up subject's 3D volumes to save space
    rm ${tmp_dir}/DR_comp_*.nii.gz
done

# Clean up temp directory
rm -rf ${tmp_dir}

echo "Done! Results saved in ${output}"
