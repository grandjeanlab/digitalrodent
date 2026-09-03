#!/bin/sh bash

# author: Joanes Grandjean
# intial date: 05.05.2022
# last modified: 30.04.2026 (Danny)

# changelog
# 30.04.2026
# Made scripts more universal, .nii.gz and .nii, TR single digit or with decimal,
# 21.04.2026
# Specify tmp scratch space
# 19.04.2026
# added automatic root_dir selection depending on where .sh lives
# 13.04.2026
# simplified the script for digitalrodent project
# 8.04.2026 
# add rabies_version to make rabies version selection more explicit
# 29.05.2024
# use the --inclusion_ids flag to run rabies one func at a time
# use the $TMPDIR environment variable to run rabies on /scratch and not on local project folder
# 03.08.2024
# modify it to run on the awake project. 


#define what root dir you want to use, where the bids folder is, where the tmp scripts will go, and where the output will go
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bids=$root_dir"/bids"
script_dir=$root_dir"/tmp_rabies_scripts_mouse"
output_dir=$root_dir"/output_mouse"
template_dir="/project/4180000.73/template/mouse"

#define what version of rabies you want to use. run `ls /opt/rabies/` to see what versions are on
rabies_version=0.6.0
rabies="/opt/rabies/${rabies_version}/rabies.sif"

#arguments for RABIES preprocessing, confound regression, analysis. see https://rabies.readthedocs.io/ for more info
prep_arg=' --bold_autobox --bold_inho_cor method=N4_reg,otsu_thresh=2,multiotsu=false --commonspace_resampling 0.2x0.2x0.2 --anatomical_resampling 0.2x0.2x0.2 --detect_dummy --oblique2card 3dWarp --commonspace_reg masking=false,brain_extraction=false,template_registration=SyN,fast_commonspace=true --anat_template '${template_dir}'/template.nii.gz --brain_mask '${template_dir}'/mask.nii.gz --WM_mask '${template_dir}'/wm.nii.gz --CSF_mask '${template_dir}'/csf.nii.gz --vascular_mask '${template_dir}'/csf.nii.gz --TR '


conf_arg_gen=' --smoothing_filter 0.3 --highpass 0.01 --lowpass 0.1 --read_datasink'

conf_arg_denoise=' --nuisance_regressors mot_6 WM_signal CSF_signal --frame_censoring FD_censoring=true,FD_threshold=0.5,DVARS_censoring=false,minimum_timepoint=3'


analysis_arg='--seed_list '${template_dir}'/s1_r.nii.gz '${template_dir}'/s1_l.nii.gz '${template_dir}'/aca_r.nii.gz '${template_dir}'/vpm_r.nii.gz --ROI_labels_file /home/traaffneu/joagra/code/awake/assets/template/mouse/labels.nii.gz --FC_matrix --prior_maps /home/traaffneu/joagra/code/awake/assets/template/mouse/ica.nii.gz --DR_ICA --prior_bold_idx 1 2 --prior_confound_idx 3 4 --ROI_labels_file '${template_dir}'/labels.nii.gz --FC_matrix --prior_maps '${template_dir}'/ica.nii.gz --DR_ICA --prior_bold_idx 1 2 --prior_confound_idx 3 4 --data_diagnosis'

#make the script directory. this is where your runnable rabies script per func scan will be run. 
mkdir -p $script_dir

mkdir -p $output_dir

cd $script_dir

#this is the main loop. by default, it will loop over every func scan that you have in your bids directory and make a separate script for it. 
find "$bids" -name '*_bold.nii*' | while read -r line
do

#need to find the corresponding json to find tr, extract it from json, and div by 1000 to get val in sec.
json=$(echo $line | sed -E 's/\.nii(\.gz)?$/.json/')
tr=$(grep "RepetitionTime" $json | sed -r 's/"RepetitionTime"://g' | sed "s/[^0-9.]//g")
#trcor=`echo $tr / 1000 | bc -l`  #because the json TR values are given in s insead of ms
trcor=$tr

#check for anatomical scans
anat_folder=$(dirname $line)
anat_scan=$(dirname $anat_folder)'/anat/*.nii.*'
bold_only=''
if [ ! -f $anat_scan ]; then
    bold_only=' --bold_only'
fi



#edit the func file name and path for rabies
##replace the full path to the bids directory with a relative path for rabies
func_file=$line


##set the name of the script file that will be created. 
func_base=$(basename $func_file)
func_noext="$(remove_ext $func_base)"
script_file=$script_dir/$func_noext'.sh'

echo "now doing subject "$func_noext

#initialize the script with a bang and slurm header. you can edit the time and mem options if you think you need more or less resources. 
echo '#!/bin/bash' > $script_file
echo "#SBATCH --job-name="$func_noext >> $script_file
echo "#SBATCH --nodes=1" >> $script_file
echo "#SBATCH --time=12:00:00" >> $script_file
echo "#SBATCH --mail-type=FAIL" >> $script_file
echo "#SBATCH --partition=batch" >> $script_file
echo "#SBATCH --mem=24GB" >> $script_file
echo "#SBATCH --tmp=75G" >> $script_file

#create temporary folders in scratch folder so you don't clutter your project folder
echo " " >> $script_file
echo "" >> $script_file
echo " " >> $script_file
echo "#### init varibles and make tmp directories ####" >> $script_file
echo " " >> $script_file

echo "preprocess=$""TMPDIR/preprocess" >> $script_file

echo "confound=$""TMPDIR/confound" >> $script_file
echo "analysis=$""TMPDIR/analysis" >> $script_file

echo "mkdir -p $""preprocess" >> $script_file

echo "mkdir -p $""confound" >> $script_file
echo "mkdir -p $""analysis" >> $script_file 


echo " " >> $script_file
echo "#### run RABIES preprocess ####" >> $script_file
echo " " >> $script_file


#run the preprocessing step of rabies
echo "apptainer run "${rabies}" --inclusion_ids "${func_file}" -p Linear preprocess "${bids}" $""{preprocess} "${bold_only} ${prep_arg}${trcor} >> $script_file

#copy the QC report, motion, and tSNR maps
echo "cp -r $""{preprocess}/preprocess_QC_report "$output_dir >> $script_file 
echo "cp -r $""{preprocess}/motion_datasink "$output_dir >> $script_file 
echo "cp -r $""{preprocess}/bold_datasink/tSNR_map_preprocess "$output_dir >> $script_file 


echo " " >> $script_file
echo "#### run RABIES confound/analysis for white matter / csf regression ####" >> $script_file
echo " " >> $script_file

#run the confound correction step of rabies
echo "apptainer run "${rabies}" --inclusion_ids "${func_file}" -p Linear confound_correction $""{preprocess} $""{confound} "${conf_arg_gen}${conf_arg_denoise} >> $script_file 
#run the analysis step of rabies
echo "apptainer run "${rabies}" --inclusion_ids "${func_file}" -p Linear analysis $""{confound} $""{analysis} "${analysis_arg} >> $script_file 
#copy the analysis outputs and the data diagnosis to the output directory
echo "cp -r $""confound/confound_correction_datasink/ "$output_dir"" >> $script_file 
echo "cp -r $""analysis/commonspace_analysis_datasink "$output_dir"" >> $script_file 
echo "cp -r $""analysis/data_diagnosis_datasink "$output_dir"" >> $script_file 


echo " " >> $script_file
echo "#### clean up####" >> $script_file
echo " " >> $script_file

#clean up scratch
echo "rm -rf $""TMPDIR/*" >> $script_file 


#uncomment one of the following if you want to run the scripts automatically (do so if you are confident it will work)
##this is if you are using the new slurm system
#sbatch $script_file

#end of the loop
done
