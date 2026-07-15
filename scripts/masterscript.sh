Template 

ask what you want to do (what do i even mean by this)

    ask if you want to do the normative model aswell 
    ask which version of pcntoolkit you would like 
    ask how many rescues you want for the rabies model
    ask/reminder if the images are all correctly oriented 

saves configuration to a file

load correct configuration for what is needed 

looks for datasets , list them ask if these are correct
creates relevant output directories for the data and results for each dataset
makes data.tsv with each dataset present 

starts creating first script for subjects 
    saves them at the correct location

when the scripts with the first parameters are created needs to ask if the user wants to run them now or later (or can be configured to run automatically)
run scripts
when there are all run it will wait for input from the user to continue to the next step
this step is manually check the QC reports and make sure everything looks good
    this needs to be done for every subject and for every dataset

    per dataset it will create a list of labels 
    the script opens the correct QC report and waits for user input to label the subject as good or bad

    if bad the script will create new parameters for rabies to run for the subject that were labeled as bad 

then continue to the next dataset and repeat the process until all datasets are completed

when all datasets are completed all subjects labeled bad will be run again with the new parameters
then continue with other parameters if they are still wrong 

There needs to be a way for the script to save and pause and even quite when neccesary 

if configured to do the normative model
run the normative model creation 
output some prelimnary results 


importants
it needs to load the correct modules on the hpc 
it needs the right directories 
it needs to succefully submit jobs to the hpc and check if they are running or completed

#!/bin/bash

###############################################################################
# MASTER SCRIPT
#
# Purpose:
# Run the different parts of the analysis in the correct order.
#
###############################################################################


###############################################################################
# BLOCK 1: GENERAL PROJECT SETTINGS
#
# These are the main locations used by the master script.
#
# Change ROOT_DIR if this script is not placed in the main project directory.
###############################################################################

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scripts_dir="$root_dir/scripts"
progress_dir="$root_dir/progress"
log_dir="$root_dir/logs"
output_dir="$root_dir/output"
tmp_rabies_scripts_dir="$root_dir/tmp_rabies"

mkdir -p "$progress_dir"
mkdir -p "$log_dir"
mkdir -p "$tmp_rabies_scripts_dir"
mkdir -p "$output_dir"

###############################################################################
# BLOCK 2: SIMPLE LOGGING
#
# This writes messages to the screen and to a log file.
###############################################################################

master_log="$log_dir/master_script.log"

log_message() {
    echo "[$(date '+%y-%m-%d %h:%m:%s')] $1" | tee -a "$master_log"
}


###############################################################################
# BLOCK 3: CHOOSE WHAT THE PIPELINE SHOULD DO
###############################################################################

run_rabies=true
do_qc=true
do_rescue=true
extract_dr_results=true
run_normative_model=false

###############################################################################
# BLOCK 4: LOCATING DATA
#
# This block identifies the datasets and checks whether the BIDS directory exists 
# for the datasets and whether it contains any subjects.
################################################################################

log_message "Looking for datasets in: $root_dir"
dataset_tsv="$progress_dir/datasets.tsv"
dataset_tsv_tmp="$progress_dir/datasets.tsv.tmp"

echo -e "dataset_name\tdataset_dir\tsubject_count" > "$dataset_tsv_tmp"

for dataset_dir in "$root_dir"/*; do

    [ -d "$dataset_dir" ] || continue

    bids_dir="$dataset_dir/bids"

    if [ -d "$bids_dir" ]; then

        dataset_name="$(basename "$dataset_dir")"

        subject_count=$(
            find "$bids_dir" \
                -maxdepth 1 \
                -type d \
                -name "sub-*" \
                | wc -l
        )

        echo -e "${dataset_name}\t${dataset_dir}\t${subject_count}" \
            >> "$dataset_tsv_tmp"
    fi

done

#mv "$dataset_tsv_tmp" "$dataset_tsv"

log_message "Datasets found:"

cat "$dataset_tsv_tmp" 
cat "$dataset_tsv_tmp" >> "$master_log"

###############################################################################
# BLOCK 5: CREATE THE RABIES JOB SCRIPTS
#
# This runs your first script.
#
# That script looks through the BIDS directory and creates one temporary
# SLURM script for every functional scan.
#
# It does not yet submit the jobs unless submission is included in that script.
###############################################################################

if [ "$run_rabies" = true ]; then


    log_message "Creating the RABIES TMP job scripts."

    bash "$scripts_dir/rabies_mouse.sh"

    if [ $? -ne 0 ]; then
        log_message "ERROR: The RABIES job scripts could not be created."
        exit 1
    fi

    log_message "RABIES job scripts were created."

fi


###############################################################################
# BLOCK 6: SUBMIT THE RABIES JOBS TO THE HPC
#
# This block will run a separate submission script.
#
# The submission script should loop through the temporary scripts and run:
#
#     sbatch temporary_script.sh
#
# For now, this block is left inactive until the submission script is ready.
###############################################################################

# log_message "Submitting the RABIES jobs."
#
# bash "$SCRIPTS_DIR/submit_rabies_jobs.sh"
#
# if [ $? -ne 0 ]; then
#     log_message "ERROR: The RABIES jobs could not be submitted."
#     exit 1
# fi
#
# log_message "RABIES jobs were submitted."


###############################################################################
# BLOCK 7: WAIT UNTIL THE HPC JOBS ARE FINISHED
#
# Eventually, this block should check whether the submitted jobs are:
#
# - waiting
# - running
# - completed
# - failed
#
# For the first version, the script simply pauses and asks the user whether
# the jobs are finished.
###############################################################################

if [ "$run_rabies" = true ]; then

    echo
    echo "The RABIES job scripts have been created."
    echo
    echo "Submit and check the jobs on the HPC."
    echo

    read -r -p "Are all preprocessing jobs finished? Type yes to continue: " answer

    if [ "$answer" != "yes" ]; then
        log_message "Pipeline stopped before QC."
        log_message "Run the master script again when the jobs are finished."
        exit 0
    fi

fi


###############################################################################
# BLOCK 8: QUALITY CONTROL
#
# This runs your QC script.
#
# The QC script:
#
# - opens one QC image
# - asks whether it is good, bad, or maybe
# - saves the answer in a CSV file
# - continues to the next subject
#
# If the user quits during QC, the master script stops.
###############################################################################

if [ "$do_qc" = true ]; then

    log_message "Starting the QC review."

    bash "$scripts_dir/review_qc.sh"

    QC_EXIT_CODE=$?

    if [ "$QC_EXIT_CODE" -ne 0 ]; then
        log_message "QC was stopped or failed."
        log_message "Run the master script again to continue later."
        exit 0
    fi

    log_message "QC review was completed."

fi


###############################################################################
# BLOCK 9: CREATE NEW JOBS FOR BAD SUBJECTS
#
# This block will eventually:
#
# 1. Read the QC labels CSV.
# 2. Find all subjects labelled as bad.
# 3. Create new temporary RABIES scripts for those subjects.
# 4. Use different RABIES parameters for the rescue attempt.
#
# This part still needs its own rescue-script creator.
###############################################################################

if [ "$do_rescue" = true ]; then

    log_message "Checking whether subjects need rescue processing."

    # This script does not exist yet.
    # It will eventually create new scripts only for bad subjects.

    # bash "$scripts_dir/create_rescue_jobs.sh"

    # if [ $? -ne 0 ]; then
    #     log_message "ERROR: Rescue jobs could not be created."
    #     exit 1
    # fi

    log_message "Rescue-job creation is not active yet."

fi


###############################################################################
# BLOCK 10: SUBMIT AND CHECK THE RESCUE JOBS
#
# Later this block should:
#
# - submit rescue jobs
# - wait until they are completed
# - start another QC round for the rescued subjects
#
# This is not active yet.
###############################################################################

# bash "$scripts_dir/submit_rescue_jobs.sh"
# bash "$scripts_dir/wait_for_rescue_jobs.sh"
# bash "$scripts_dir/review_rescue_qc.sh"


###############################################################################
# BLOCK 11: EXTRACT THE DUAL-REGRESSION RESULTS
#
# This runs your third script.
#
# It extracts the beta and Z-score values and saves them as CSV files.
###############################################################################

if [ "$extract_dr_results" = true ]; then

    echo
    read -r -p "Are all subjects processed and approved by QC? Type yes to extract the results: " answer

    if [ "$answer" = "yes" ]; then

        log_message "Extracting the dual-regression results."

        bash "$scripts_dir/extract_dual_regression.sh"

        if [ $? -ne 0 ]; then
            log_message "ERROR: Dual-regression results could not be extracted."
            exit 1
        fi

        log_message "Dual-regression results were extracted."

    else
        log_message "Dual-regression extraction was skipped."
    fi

fi


###############################################################################
# BLOCK 12: NORMATIVE MODEL
#
# This block is only run when RUN_NORMATIVE_MODEL is set to true.
#
# The exact normative-model script can be added later.
###############################################################################

if [ "$run_normative_model" = true ]; then

    log_message "Starting the normative model."

    bash "$scripts_dir/run_normative_model.sh"

    if [ $? -ne 0 ]; then
        log_message "ERROR: The normative model failed."
        exit 1
    fi

    log_message "Normative model completed."

else

    log_message "Normative modelling is turned off."

fi


###############################################################################
# BLOCK 13: FINISH
###############################################################################

log_message "The master script has reached the end."

echo
echo "Finished."
echo "Check the log file here:"
echo "$master_log"