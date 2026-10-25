# Model Encoding


``` python
from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

current_dir = Path.cwd().parent

input_path = current_dir / "results" / "combined_dataset" / "Combined_dataset.csv"

output_dir = current_dir / "results" / "combined_dataset"

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(input_path)

print("Original rows:", len(df))


# ============================================================
# WANTED VARIABLES ONLY
# ============================================================

id_columns = [
    "Scan_id",
    "Participant_Id",
    "Session",
    "Dataset_name"
]

response_vars = [
    f"comp{i}" for i in range(1, 19)
]

numeric_covariates = [
    "Age(week)",
    "MRI",
    "MRI T.R.",
    "FD average",   # motion, per run
    "log_tSNR"      # signal quality, per run (log-transformed below)
]

# Per-run quality measures: never copied from another run of the same animal
per_scan_covariates = ["FD average", "log_tSNR"]

categorical_covariates = [
    "Sex",
    "Rodent.strain",
    "Coil",
    "fMRI.sequence",
    "Anesthesia",
    "Ventilation"
]

# tSNR is strongly right-skewed (median ~17, max ~240): use log(tSNR)
df["log_tSNR"] = np.log(pd.to_numeric(df["tSNR"], errors="coerce"))

wanted_columns = (
    id_columns
    + numeric_covariates
    + categorical_covariates
    + response_vars
)

df = df[wanted_columns].copy()

print("Columns kept:")
print(df.columns.tolist())


# ============================================================
# CLEAN BASIC VALUES
# ============================================================

for col in numeric_covariates + response_vars:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

for col in categorical_covariates:
    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
    )

    df[col] = df[col].replace({
        "": pd.NA,
        "NA": pd.NA,
        "N/A": pd.NA,
        "Missing": pd.NA,
        "missing": pd.NA,
        "MISSING": pd.NA
    })


# ============================================================
# STANDARDIZE CATEGORIES
# ============================================================

df["Sex"] = (
    df["Sex"]
    .str.upper()
    .replace({
        "FEMALE": "F",
        "MALE": "M"
    })
)

df["Rodent.strain"] = (
    df["Rodent.strain"]
    .replace({
        "C57BL6": "C57BL/6",
        "C57BL6J": "C57BL/6",
        "C57BL/6J": "C57BL/6"
    })
)

df["Coil"] = (
    df["Coil"]
    .str.lower()
    .replace({
        "rt": "RT",
        "cryo": "Cryo",
        "n": "RT",
        "no": "RT",
        "y": "Cryo",
        "yes": "Cryo"
    })
)

df["fMRI.sequence"] = (
    df["fMRI.sequence"]
    .str.upper()
    .replace({
        "EPI": "GE-EPI",
        "GE EPI": "GE-EPI",
        "SE EPI": "SE-EPI",
        "ME EPI": "ME-EPI"
    })
)

df["Anesthesia"] = (
    df["Anesthesia"]
    .str.lower()
    .replace({
        "awake": "Awake",
        "iso": "Iso",
        "dex": "Dex",
        "dexiso": "Dexiso",
        "med": "Med",
        "mediso": "Mediso",
        "hal": "Hal",
        "propofol": "Propofol"
    })
)

df["Ventilation"] = (
    df["Ventilation"]
    .str.lower()
    .replace({
        "n": "No",
        "no": "No",
        "y": "Yes",
        "yes": "Yes"
    })
)


# ============================================================
# REMOVE VERY RARE CATEGORIES
# ICR strain (4 scans) and Dex anesthesia (5 scans): with so few
# scans their coefficients are unstable and they make the
# heteroskedastic variance model blow up (|z| > 50 in testing)
# ============================================================

rare = (df["Rodent.strain"] == "ICR") | (df["Anesthesia"] == "Dex")
print("\nRemoved rare categories:", int(rare.sum()), "scans")
print(df.loc[rare].groupby(["Dataset_name", "Rodent.strain", "Anesthesia"]).size())
df = df.loc[~rare].copy()


# ============================================================
# MISSING VALUES BEFORE INFERENCE
# ============================================================

required_columns = (
    numeric_covariates
    + categorical_covariates
    + response_vars
)

missing_before = (
    df[required_columns]
    .isna()
    .sum()
)


# ============================================================
# INFER FROM SAME PARTICIPANT
# ============================================================

inferable_columns = [
    col for col in numeric_covariates + categorical_covariates
    if col not in per_scan_covariates
]


def fill_if_unique(series):
    observed = (
        series
        .dropna()
        .unique()
    )

    if len(observed) == 1:
        return series.fillna(observed[0])

    return series


for col in inferable_columns:
    df[col] = (
        df
        .groupby(
            ["Participant_Id", "Dataset_name"],
            dropna=False
        )[col]
        .transform(fill_if_unique)
    )


# ============================================================
# MISSING VALUES AFTER INFERENCE
# ============================================================

missing_after = (
    df[required_columns]
    .isna()
    .sum()
)

inferred_counts = (
    missing_before
    - missing_after
)


# ============================================================
# SEX STILL UNKNOWN -> OWN CATEGORY
# Sex cannot be recovered for part of Multicenter. Instead of
# deleting these scans (and losing e.g. all medetomidine-only
# scans), unknown sex becomes its own category, modelled with
# an indicator (Sex_unknown) next to Sex_encoded.
# ============================================================

df["Sex"] = df["Sex"].fillna("Unknown")

print("\nSex categories after filling:")
print(df.groupby(["Dataset_name", "Sex"]).size().unstack(fill_value=0))


# ============================================================
# DELETE ROWS STILL MISSING WANTED MODEL VALUES
# ============================================================

delete_mask = (
    df[required_columns]
    .isna()
    .any(axis=1)
)

deleted_rows = df.loc[
    delete_mask
].copy()

model_df = df.loc[
    ~delete_mask
].copy()


# ============================================================
# DELETION SUMMARY
# ============================================================

deletion_summary = pd.DataFrame({
    "Variable": required_columns,
    "Missing_before_inference": [
        missing_before[col]
        for col in required_columns
    ],
    "Inferred_from_participant": [
        inferred_counts[col]
        for col in required_columns
    ],
    "Missing_after_inference": [
        missing_after[col]
        for col in required_columns
    ]
})

print("\nMissing-value summary:")
display(deletion_summary)

print("\nRows before:", len(df))
print("Rows deleted:", delete_mask.sum())
print("Rows retained:", len(model_df))

print("\nDeleted rows by dataset:")
display(
    deleted_rows["Dataset_name"]
    .value_counts()
    .rename_axis("Dataset_name")
    .reset_index(name="N_deleted")
)


# ============================================================
# ENCODE BINARY VARIABLES
# ============================================================

# F = reference; unknown sex gets Sex_encoded = 0 and Sex_unknown = 1
model_df["Sex_encoded"] = (
    model_df["Sex"]
    .map({
        "F": 0.0,
        "M": 1.0,
        "Unknown": 0.0
    })
)

model_df["Sex_unknown"] = (
    model_df["Sex"] == "Unknown"
).astype(float)

model_df["Coil_encoded"] = (
    model_df["Coil"]
    .map({
        "RT": 0.0,
        "Cryo": 1.0
    })
)

model_df["Ventilation_encoded"] = (
    model_df["Ventilation"]
    .map({
        "No": 0.0,
        "Yes": 1.0
    })
)

encoding_table = [
    ["Sex", "F", "0"],
    ["Sex", "M", "1"],
    ["Sex", "Unknown", "Sex_unknown = 1"],
    ["Coil", "RT", "0"],
    ["Coil", "Cryo", "1"],
    ["Ventilation", "No", "0"],
    ["Ventilation", "Yes", "1"]
]


# ============================================================
# DUMMY ENCODE MULTI-CATEGORY VARIABLES
# ============================================================

dummy_settings = {
    "Rodent.strain": "C57BL/6",
    "fMRI.sequence": "GE-EPI",
    "Anesthesia": "Awake"
}

dummy_columns = []


for col, reference in dummy_settings.items():

    categories = sorted(
        model_df[col]
        .dropna()
        .unique()
        .tolist()
    )

    if reference not in categories:
        raise ValueError(
            f"Reference category {reference} not found in {col}"
        )

    ordered_categories = (
        [reference]
        + [
            x for x in categories
            if x != reference
        ]
    )

    model_df[col] = pd.Categorical(
        model_df[col],
        categories=ordered_categories
    )

    dummies = pd.get_dummies(
        model_df[col],
        prefix=col,
        drop_first=True,
        dtype=float
    )

    model_df = pd.concat(
        [model_df, dummies],
        axis=1
    )

    dummy_columns.extend(
        dummies.columns.tolist()
    )

    encoding_table.append([
        col,
        reference,
        "Reference"
    ])

    for dummy in dummies.columns:
        category = dummy.replace(
            col + "_",
            ""
        )

        encoding_table.append([
            col,
            category,
            dummy
        ])


# ============================================================
# FINAL MODEL COVARIATES
# ============================================================

covariates = (
    numeric_covariates
    + [
        "Sex_encoded",
        "Sex_unknown",
        "Coil_encoded",
        "Ventilation_encoded"
    ]
    + dummy_columns
)


# ============================================================
# FINAL DATAFRAME
# ============================================================

final_columns = (
    id_columns
    + covariates
    + response_vars
)

partial_model_df = (
    model_df[final_columns]
    .copy()
)


# ============================================================
# FINAL CHECKS
# ============================================================

remaining_missing = (
    partial_model_df[
        covariates
        + response_vars
    ]
    .isna()
    .sum()
    .sum()
)

if remaining_missing != 0:
    raise ValueError(
        f"{remaining_missing} missing values remain."
    )

encoding_table = pd.DataFrame(
    encoding_table,
    columns=[
        "Variable",
        "Category",
        "Encoding"
    ]
)


# ============================================================
# SAVE
# ============================================================

# ============================================================
# SAVE
# ============================================================

partial_model_df.to_csv(
    output_dir / "Model_dataframe.csv",
    index=False
)

encoding_table.to_csv(
    output_dir / "Encoding_table.csv",
    index=False
)

deletion_summary.to_csv(
    output_dir / "Deletion_summary.csv",
    index=False
)

deleted_rows.to_csv(
    output_dir / "Deleted_rows.csv",
    index=False
)
```

    Original rows: 2153
    Columns kept:
    ['Scan_id', 'Participant_Id', 'Session', 'Dataset_name', 'Age(week)', 'MRI', 'MRI T.R.', 'FD average', 'log_tSNR', 'Sex', 'Rodent.strain', 'Coil', 'fMRI.sequence', 'Anesthesia', 'Ventilation', 'comp1', 'comp2', 'comp3', 'comp4', 'comp5', 'comp6', 'comp7', 'comp8', 'comp9', 'comp10', 'comp11', 'comp12', 'comp13', 'comp14', 'comp15', 'comp16', 'comp17', 'comp18']

    Removed rare categories: 9 scans
    Dataset_name  Rodent.strain  Anesthesia
    Awake         ICR            Awake         4
    Douglas       C57BL/6        Dex           5
    dtype: int64

    Sex categories after filling:
    Sex                 F    M  Unknown
    Dataset_name                       
    3xTG                0   21        0
    Alvino              8   14        0
    Awake             661  773        0
    Awake_Rest        166    0        0
    Douglas             0   15        0
    Env_Enrichment     22   16        0
    Florida             6    6        0
    Longitudinal       27   27        0
    MPTP                0    9        0
    Multicenter        35   67      139
    NDUFS4             10   18        0
    Propofol            0    0        5
    Psilocybin          0   15        0
    Psilocybin_Pilot    0   24        0
    Sexdifference      30   30        0

    Missing-value summary:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Variable | Missing_before_inference | Inferred_from_participant | Missing_after_inference |
|----|----|----|----|----|
| 0 | Age(week) | 222 | 0 | 222 |
| 1 | MRI | 0 | 0 | 0 |
| 2 | MRI T.R. | 0 | 0 | 0 |
| 3 | FD average | 2 | 0 | 2 |
| 4 | log_tSNR | 0 | 0 | 0 |
| 5 | Sex | 144 | 0 | 144 |
| 6 | Rodent.strain | 0 | 0 | 0 |
| 7 | Coil | 0 | 0 | 0 |
| 8 | fMRI.sequence | 0 | 0 | 0 |
| 9 | Anesthesia | 0 | 0 | 0 |
| 10 | Ventilation | 0 | 0 | 0 |
| 11 | comp1 | 0 | 0 | 0 |
| 12 | comp2 | 0 | 0 | 0 |
| 13 | comp3 | 0 | 0 | 0 |
| 14 | comp4 | 0 | 0 | 0 |
| 15 | comp5 | 0 | 0 | 0 |
| 16 | comp6 | 0 | 0 | 0 |
| 17 | comp7 | 0 | 0 | 0 |
| 18 | comp8 | 0 | 0 | 0 |
| 19 | comp9 | 0 | 0 | 0 |
| 20 | comp10 | 0 | 0 | 0 |
| 21 | comp11 | 0 | 0 | 0 |
| 22 | comp12 | 0 | 0 | 0 |
| 23 | comp13 | 0 | 0 | 0 |
| 24 | comp14 | 0 | 0 | 0 |
| 25 | comp15 | 0 | 0 | 0 |
| 26 | comp16 | 0 | 0 | 0 |
| 27 | comp17 | 0 | 0 | 0 |
| 28 | comp18 | 0 | 0 | 0 |

</div>

    Rows before: 2144
    Rows deleted: 224
    Rows retained: 1920

    Deleted rows by dataset:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | Dataset_name | N_deleted |
|-----|--------------|-----------|
| 0   | Awake        | 219       |
| 1   | Propofol     | 5         |

</div>
