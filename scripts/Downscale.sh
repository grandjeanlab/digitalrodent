#!/usr/bin/env bash
# =============================================================================
# Sometimes you find datasets where the NIfTI voxel size is off by a factor
# of 10 compared to the raw acquisition. To run RABIES you need
# to downscale pixdim back to the real size.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------
# STEP 0 - single subject, manual example (check + safety copy + rescale)
# -----------------------------------------------------------------------
# fslinfo your_image.nii.gz                            # check current dims
# cp your_image.nii.gz your_image_backup.nii.gz         # safety copy
# fslchpixdim your_image_backup.nii.gz 0.2 0.2 0.6       # rescale by 10 using your current dims values (example values) 

# -----------------------------------------------------------------------
# STEP 1 - loop: rescale pixdim for all subjects
# -----------------------------------------------------------------------
for nii in sub-*/func/*.nii.gz; do
    echo "Adjusting pixdim for: $nii"
    fslchpixdim "$nii" 0.2 0.2 0.6
done
echo "Pixdim adjustment complete for all subjects!"

# -----------------------------------------------------------------------
# STEP 2 - loop: check qform/sform mismatch, fix only if needed
# -----------------------------------------------------------------------
for f in sub-*/func/*_bold.nii.gz; do
    q=$(fslhd "$f" | grep "^qto_xyz:2" | awk '{$1=""; print}')
    s=$(fslhd "$f" | grep "^sto_xyz:2" | awk '{$1=""; print}')

    if [ "$q" != "$s" ]; then
        echo "MISMATCH found, fixing: $f"
        fslorient -copyqform2sform "$f"
    else
        echo "OK, no mismatch: $f"
    fi
done

# -----------------------------------------------------------------------
# STEP 3 - verification
# -----------------------------------------------------------------------
for f in sub-*/func/*_bold.nii.gz; do
    echo "=== $f ==="
    fslhd "$f" | grep -A4 sto_xyz
done