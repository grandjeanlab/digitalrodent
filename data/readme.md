# Datasets

This directory contains metadata describing the datasets used in the Digital Rodent project.

## Contents

### `datasets.tsv`

A tab-separated table containing information about each dataset included in this project. The table provides a unique dataset identifier (`dataset_id`) together with descriptive information such as the dataset name, source, DOI (where available), and other relevant metadata.

The `dataset_id` is used consistently throughout this repository to link datasets with processed results and analyses.

## Raw data

The raw MRI datasets are **not** included in this repository.

Several datasets originate from publicly available repositories (e.g. OpenNeuro), while others are shared through collaborations or are subject to data-sharing restrictions. Users should obtain these datasets directly from their original source or the corresponding data providers.

## Processed results

Processed outputs derived from these datasets (where redistribution is permitted) are stored separately in the `results/` directory. Files are linked to the corresponding dataset through the `dataset_id` defined in `datasets.tsv`.

## Adding a new dataset

When adding a new dataset:

1. Assign the next available `dataset_id`.
2. Add a new row to `datasets.tsv`.
3. Include the dataset name, source, DOI (if available), and any other relevant metadata.
4. Use the assigned `dataset_id` consistently for processed outputs stored in the `results/` directory.

# Subjects 

Not all subjects from all datasets were compatible with our analysis, for the normative model healthy and control subjects were used to create an accurate model. Other animals did not deliver correct results and the image data was just not processable to use for input of the model.

## 'subject.tsv' 
shows the subjects that were included in the normative model

## 
