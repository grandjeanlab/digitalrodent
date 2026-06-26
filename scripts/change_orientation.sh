#!/usr/bin/env bash
for nii in bids/sub*/func/*.ni*; do
  echo "Processing $nii"

  dir=$(dirname "$nii")
  file=$(basename "$nii")

  tmp="${dir}/${base}_tmp.nii.gz"
  out="${dir}/${base}_fixed.nii.gz"

  # Process
  fslorient -deleteorient "$nii"
  fslswapdim "$nii" x z y "$tmp"
  3dresample -input "$tmp" -prefix "$out" -orient ras

done

#!/usr/bin/env bash
for nii in bids/sub*/anat/*.ni*; do
  echo "Processing $nii"

  dir=$(dirname "$nii")
  file=$(basename "$nii")

  tmp="${dir}/${base}_tmp.nii.gz"
  out="${dir}/${base}_fixed.nii.gz"

  # Process
  fslorient -deleteorient "$nii"
  fslswapdim "$nii" x z y "$tmp"
  3dresample -input "$tmp" -prefix "$out" -orient ras

done
