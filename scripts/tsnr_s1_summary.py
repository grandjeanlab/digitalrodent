"""
Mean tSNR in bilateral S1 for every scan preprocessed with RABIES.

What it does
------------
RABIES writes one temporal SNR map per scan (tSNR = mean / std of the BOLD
signal over time, voxel-wise) to <output>/tSNR_map_preprocess/. This script:

  1. finds every tSNR map in that folder;
  2. averages the tSNR inside a bilateral S1 mask (one value per scan),
     using a nipype workflow so the maps are processed in parallel;
  3. writes one small TSV per scan to the datasink;
  4. collects them into a single summary table, one row per scan:

        subject  session  task  run  s1_bilateral  file

Use the summary to spot scans with poor signal in S1 (e.g. to set an
exclusion threshold or compare groups/sessions).

Requirements
------------
- The mask must be in the same space and resolution as the tSNR maps
  (RABIES commonspace, i.e. the template used for preprocessing).
- The mask is binary (0/1). Build it once from the left and right S1 ROIs:
      fslmaths s1_l.nii.gz -add s1_r.nii.gz -bin s1_bilateral.nii.gz
"""

import re
from pathlib import Path

import pandas as pd
from nipype import Function, IdentityInterface, Node, Workflow
from nipype.interfaces.io import DataFinder, DataSink
from nipype.interfaces.nilearn import SignalExtraction

# ---- config ----------------------------------------------------------------
BASE_DIR = Path('/project/4180000.73/Propofol_anethesia_dataset')
TSNR_DIR = BASE_DIR / 'output_mouse' / 'tSNR_map_preprocess'   # RABIES output
OUT_DIR = BASE_DIR / 'analysis_mouse_tSNR'

MASK_FILE = '/project/4180000.73/template/mouse/s1_bilateral.nii.gz'
MASK_NAME = 's1_bilateral'
N_PROCS = 4

WORK_DIR = OUT_DIR / f'workflow_{MASK_NAME}'          # nipype cache
SINK_DIR = OUT_DIR / 'datasink' / f'tsnr_{MASK_NAME}'  # per-scan TSVs

SUMMARY_FILE = OUT_DIR / 'datasink' / f'tsnr_{MASK_NAME}_summary.tsv'

BIDS_RE = re.compile(
    r'(?P<subject>sub-[A-Za-z0-9]+)'
    r'(?:_(?P<session>ses-[A-Za-z0-9]+))?'
    r'(?:_task-(?P<task>[A-Za-z0-9]+))?'
    r'(?:_run-(?P<run>[A-Za-z0-9]+))?'
)


def make_filename(in_file):
    """Per-scan TSV name from the map's path relative to tSNR_map_preprocess/.

    Must be self-contained: nipype Function nodes don't see module-level
    imports or variables.
    """
    rel = in_file.split('tSNR_map_preprocess/')[1].replace('.nii.gz', '')
    return rel.replace('/', '_') + '.tsv'


def find_tsnr_maps(root):
    """List all .nii.gz files under root (always returns a list)."""
    finder = DataFinder(root_paths=str(root),
                        match_regex=r'^(?P<basedir>.+)/(?P<filename>[^/]+)\.nii\.gz$')
    paths = finder.run().outputs.out_paths
    return [paths] if isinstance(paths, str) else sorted(paths)


def run_extraction(tsnr_maps):
    """Mean tSNR within the mask for each map, one TSV per map in SINK_DIR."""
    # one workflow branch per map
    infosource = Node(IdentityInterface(fields=['in_file']), name='infosource')
    infosource.iterables = [('in_file', tsnr_maps)]

    filename = Node(Function(input_names=['in_file'], output_names=['filename'],
                             function=make_filename),
                    name='filename')

    extract = Node(SignalExtraction(), name='extract')
    extract.inputs.label_files = MASK_FILE      # single binary mask -> one value
    extract.inputs.class_labels = [MASK_NAME]

    sink = Node(DataSink(), name='sink')
    sink.inputs.base_directory = str(SINK_DIR)
    sink.inputs.parameterization = False

    wf = Workflow(name=f'tsnr_{MASK_NAME}', base_dir=str(WORK_DIR))
    wf.connect([
        (infosource, extract, [('in_file', 'in_file')]),
        (infosource, filename, [('in_file', 'in_file')]),
        (filename, extract, [('filename', 'out_file')]),
        (extract, sink, [('out_file', 'out_file')]),
    ])
    wf.run(plugin='MultiProc', plugin_args={'n_procs': N_PROCS})


def build_summary():
    """Collect the per-scan TSVs into one table, one row per scan."""
    rows = []
    for f in sorted(SINK_DIR.rglob('*.tsv')):
        value = pd.read_csv(f, sep='\t')[MASK_NAME].mean()
        m = BIDS_RE.search(f.name)
        ids = m.groupdict(default='') if m else {'subject': f.stem}
        rows.append({**ids, MASK_NAME: value, 'file': f.name})

    cols = ['subject', 'session', 'task', 'run', MASK_NAME, 'file']
    return pd.DataFrame(rows, columns=cols)


def main():
    tsnr_maps = find_tsnr_maps(TSNR_DIR)
    print(f'Found {len(tsnr_maps)} tSNR maps in {TSNR_DIR}')

    run_extraction(tsnr_maps)

    summary = build_summary()
    summary.to_csv(SUMMARY_FILE, sep='\t', index=False)
    print(f'Summary of {len(summary)} maps saved to {SUMMARY_FILE}')


if __name__ == '__main__':
    main()
