#!/bin/bash

# author: Danny Schuurman
# intial date: 24.04.2026
# last modified: 21.04.2026 (Danny)

img_dir="/project/4180000.73/University_of_Florida_Mouse_Sepsis_fMRI_StudyT/output_mouse/preprocess_QC_report/commonspace_reg_wf.Anat2Atlas/"

#Change based on rescue
log_file="labels1.csv"
rescue="sub"

# Create log file if missing
[ ! -f "$log_file" ] && echo "filename,label" > "$log_file"

Files=( "$img_dir"/*"$rescue"* )

for img in "${files[@]}"; do
    [ -e "$img" ] || continue

    # Open image in Gwenview
    gwenview "$img" &
    GWEN_PID=$!

    # Give Gwenview a moment to open
    sleep 1

    # Ask for label
    echo "Label this image: A=good, S=bad, D=maybe, q=quit"
    read -n 1 key
    echo
S
    [[ "$key" == "q" ]] && break

    # Map keys to labels
    case "$key" in
        A|a) label="good" ;;
        S|s) label="bad" ;;
        D|d) label="maybe" ;;
        *) label="unknown" ;;
    esac

    # Log to CSV
    echo "$(basename "$img"),$label" >> "$log_file"
done

echo "Good: $(grep -c ',good$' "$log_file")"
grep ',good$' "$log_file" | cut -d',' -f1

echo "Bad: $(grep -c ',bad$' "$log_file")"
grep ',bad$' "$log_file" | cut -d',' -f1

echo "Maybe: $(grep -c ',maybe$' "$log_file")"
grep ',maybe$' "$log_file" | cut -d',' -f1
