# Digital Rodent

## Analysis workflow

The `Analysis/` folder contains the Python and Quarto scripts used for data preparation, analysis, normative modeling and visualization. The scripts are organized in the order of the analysis workflow.

### 1. Clean_Merge.qmd

Combines the dual regression results from the 15 datasets with their corresponding metadata. Participant IDs and sessions are matched, metadata variables are standardized, and excluded scans are removed.

The resulting dataset is saved as `results/combined_dataset/Combined_dataset.csv`.

### 2. Data_Analysis.qmd

Performs exploratory analysis of the combined dataset. This includes dataset composition, demographic information, quality control, and the distribution of dual regression values across datasets. The script also generates figures comparing the Alzheimer and 3xTG datasets.

Figures are saved in `results/Figures_Analysis/`.

### 3. Model_Encoding.qmd

Prepares the combined dataset for normative modeling. The script selects the relevant covariates and all 18 functional components, checks missing values, and attempts to infer missing metadata from other scans of the same animal.

Scans with remaining missing values are excluded, and categorical variables are encoded for BLR.

The resulting modeling dataframe, encoding table, and deletion summaries are saved in `results/combined_dataset/`.

### 4. Model_Training.qmd

Trains a normative model using Bayesian Linear Regression (BLR) implemented in PCNtoolkit. The model includes all 18 functional components and uses biological and acquisition related variables as covariates.

The script splits the data into training and test sets, fits the model, and evaluates its performance.

The trained model, statistics, and evaluation plots are saved in `results/normative_model/`.

### 5. Model_Centiles.qmd

Loads the trained normative model and generates centile plots showing the expected functional connectivity values across different biological and acquisition variables.

Although the model includes all 18 components, the figures focus on the components listed in `SELECTED_COMPONENTS` in `Analysis/component_labels.py` (currently M1 = comp2, S1uLp + S2 = comp3, V2 + RSD = comp7, Cg1 + RS = comp10, Ventral HP = comp12).

#### Component labels

All scripts take the component names from one file, `Analysis/component_labels.py` (new ICA labels, Zerbi). Components are numbered 1-18, the parcellation 0-17: comp{n} = parcellation index n-1. Inside the model the response variables stay `comp1`-`comp18`.

| Component | Network | Component | Network |
|---|---|---|---|
| comp1 | Pir | comp10 | Cg1 + RS |
| comp2 | M1 | comp11 | Dorsal HP / superior colliculus |
| comp3 | S1uLp + S2 | comp12 | Ventral HP |
| comp4 | S1 HL + FL | comp13 | CPu |
| comp5 | S1 FL + S1 BF | comp14 | CPu + En |
| comp6 | S1 BF | comp15 | Septum + VP |
| comp7 | V2 + RSD | comp16 | Amygdala |
| comp8 | AUd + AUv/TeA-ECT | comp17 | Thalamus |
| comp9 | Cg2 | comp18 | MO + GI |

The plots are saved as SVG files in `results/normative_model/centiles_svg/`.

### 6. Model_Centiles_Alzheimer.qmd

Applies the trained normative model to an independent Alzheimer dataset. The script generates age dependent centile plots with the Alzheimer observations overlaid on the reference data.

It also calculates normative deviation scores for the Alzheimer scans, allowing their connectivity values to be compared with the expected values from the healthy reference model.

The centile figures are saved in `results/normative_model/centiles_svg_Alzheimer/`, and the calculated z scores are saved in `results/normative_model/Alzheimer_Zscores.csv`.

### 7. Model_Alzheimer_Comparison.qmd

Compares wild-type controls (`Dataset_name == "3xTG"`, the controls of the 3xTG study) with 3xTG-AD mice (`data/Alzheimer/`), on raw values and on normative deviation scores. The normative model is refit without the wild-type controls (same settings as `Model_Training.qmd`) so that both groups are unseen by the model.

Outputs: Figure 4A (overlapping group distributions, raw vs z) and Figure 4B (group difference testing, SVM classification, effect sizes) in `results/Figures_Analysis/`; the refit model, per-scan z-scores and the group-difference table in `results/normative_model_WT_heldout/`. Requires `scikit-learn` and `statsmodels`.

## Running the analysis

The scripts should be executed in the order described above. Each Quarto file can be rendered using:

```bash
quarto render Clean_Merge.qmd --to gfm
```

Rendering executes the Python code and creates a Markdown file containing both the code and its outputs.

The data preparation and exploratory analyses can be performed locally. Model training and the subsequent centile analyses can be performed on the HPC using PCNtoolkit.

The workflow uses the generated files from previous steps, so the scripts should be executed in the correct order.