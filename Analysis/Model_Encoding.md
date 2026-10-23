# Model Encoding


``` python
from pathlib import Path
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
    "MRI T.R."
]

categorical_covariates = [
    "Sex",
    "Rodent.strain",
    "Coil",
    "fMRI.sequence",
    "Anesthesia",
    "Ventilation"
]

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

inferable_columns = (
    numeric_covariates
    + categorical_covariates
)


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

model_df["Sex_encoded"] = (
    model_df["Sex"]
    .map({
        "F": 0.0,
        "M": 1.0
    })
)

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

    Original rows: 2020
    Columns kept:
    ['Scan_id', 'Participant_Id', 'Session', 'Dataset_name', 'Age(week)', 'MRI', 'MRI T.R.', 'Sex', 'Rodent.strain', 'Coil', 'fMRI.sequence', 'Anesthesia', 'Ventilation', 'comp1', 'comp2', 'comp3', 'comp4', 'comp5', 'comp6', 'comp7', 'comp8', 'comp9', 'comp10', 'comp11', 'comp12', 'comp13', 'comp14', 'comp15', 'comp16', 'comp17', 'comp18']

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
| 0 | Age(week) | 218 | 0 | 218 |
| 1 | MRI | 10 | 0 | 10 |
| 2 | MRI T.R. | 10 | 0 | 10 |
| 3 | Sex | 182 | 0 | 182 |
| 4 | Rodent.strain | 10 | 0 | 10 |
| 5 | Coil | 0 | 0 | 0 |
| 6 | fMRI.sequence | 10 | 0 | 10 |
| 7 | Anesthesia | 0 | 0 | 0 |
| 8 | Ventilation | 0 | 0 | 0 |
| 9 | comp1 | 0 | 0 | 0 |
| 10 | comp2 | 0 | 0 | 0 |
| 11 | comp3 | 0 | 0 | 0 |
| 12 | comp4 | 0 | 0 | 0 |
| 13 | comp5 | 0 | 0 | 0 |
| 14 | comp6 | 0 | 0 | 0 |
| 15 | comp7 | 0 | 0 | 0 |
| 16 | comp8 | 0 | 0 | 0 |
| 17 | comp9 | 0 | 0 | 0 |
| 18 | comp10 | 0 | 0 | 0 |
| 19 | comp11 | 0 | 0 | 0 |
| 20 | comp12 | 0 | 0 | 0 |
| 21 | comp13 | 0 | 0 | 0 |
| 22 | comp14 | 0 | 0 | 0 |
| 23 | comp15 | 0 | 0 | 0 |
| 24 | comp16 | 0 | 0 | 0 |
| 25 | comp17 | 0 | 0 | 0 |
| 26 | comp18 | 0 | 0 | 0 |

</div>


    Rows before: 2020
    Rows deleted: 386
    Rows retained: 1634

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
| 0   | Awake        | 195       |
| 1   | Multicenter  | 140       |
| 2   | NDUFS4       | 28        |
| 3   | MPTP         | 9         |
| 4   | 3xTG         | 7         |
| 5   | Propofol     | 4         |
| 6   | Awake_Rest   | 3         |

</div>
