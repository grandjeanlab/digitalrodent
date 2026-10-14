#!/bin/env bash


hcp_dir="/project_cephfs/3022017.01/S1200"
out_dir="/project/4180000.73/hcp_dr/"
template_dir="/project/4180000.73/hcp_dr/template/"
scan_pattern="hp2000_clean"

#assemble dr
#cd $template_dir/split
#ic_list=$(ls .)

#fslmerge -t ../ic_template.nii.gz $ic_list
#fslmaths ../ic_template.nii.gz -bin ../ic_template.nii.gz 



#do the analysis
cd $out_dir
mkdir -p dr

fd -e .nii.gz --exclude "*tfMRI*" $scan_pattern $hcp_dir | while read line
do

subject=$(echo "$line" | cut -d '/' -f 5)
scan=$(echo "$line" | cut -d '/' -f 8)
out_name="${subject}_${scan}"

sbatch --mem=16G --time=00:10:00 <<EOF
#!/bin/bash
dual_regression "$template_dir/ic_template.nii.gz" 1 -1 0 "dr/$out_name" "$line"
rm dr/$out_name/dr_stage2_ic*
rm dr/$out_name/dr_stage1*
rm dr/$out_name/mask*
rm -rf dr/$out_name/scripts*
EOF

sleep 5

done







