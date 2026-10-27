# Data Cleaning and Merging


``` python
from pathlib import Path
import pandas as pd
import re

root = Path.cwd().parent

meta_dir = root / "data" / "MetaData"
dr_dir = root / "results" / "dual_regression"
out_dir = root / "results" / "combined_dataset"

datasets = pd.read_csv(root / "data" / "datasets.tsv", sep="\t")


def clean_metadata_id(dataset, x):
    
    x = str(x).strip()

    if x.endswith(".0"):
        x = x[:-2]

    # Awake_Rest IDs may already contain session
    x = re.sub(r"_ses-\d+$", "", x)

    # Awake IDs have extra leading zeroes in DR
    if dataset == "Awake":
        x = x.replace("sub-", "")
        x = x.lstrip("0")

    return x.lower()


def clean_dr_id(dataset, subject):

    subject = str(subject).strip()

    # Longitudinal
    if dataset == "Longitudinal":
        return subject.replace("_ses-", "-").lower()

    # Remove session/run information
    subject = re.sub(r"_ses-\d+.*$", "", subject)

    # Datasets with _task-rest instead of session
    subject = re.sub(r"_task-.*$", "", subject)

    # 3xTG
    if dataset == "3xTG":
        subject = subject.replace("sub-jgrAD", "")

    # Awake
    if dataset == "Awake":
        subject = subject.replace("sub-", "")
        subject = subject.lstrip("0")

    return subject.lower()


def get_dr_session(dataset, subject):

    subject = str(subject)

    # Longitudinal
    if "_ses-PND30" in subject:
        return 1

    if "_ses-PND90" in subject:
        return 2

    # Normal session
    match = re.search(r"_ses-(\d+)", subject)

    if match:
        return int(match.group(1))

    # No session in scan name
    return 1
```

``` python
all_datasets = []

# Datasets left out completely
# Propofol: no age, sex, FD or tSNR for most scans -> not usable
excluded_datasets = ["Propofol"]

for dataset in datasets["Nickname"]:

    if dataset in excluded_datasets:
        print(f"Skipping dataset {dataset} (in excluded_datasets)")
        continue

    metadata_file = meta_dir / f"{dataset}.tsv"
    dr_file = dr_dir / f"{dataset}.csv"

    if not metadata_file.exists() or not dr_file.exists():
        continue

    metadata = pd.read_csv(metadata_file, sep="\t")
    dr = pd.read_csv(dr_file)

    # Create matching variables
    metadata["match_id"] = metadata["Participant_Id"].apply(
        lambda x: clean_metadata_id(dataset, x)
    )

    metadata["match_session"] = pd.to_numeric(
        metadata["Session"],
        errors="coerce"
    )

    dr["match_id"] = dr["subject"].apply(
        lambda x: clean_dr_id(dataset, x)
    )

    dr["match_session"] = dr["subject"].apply(
        lambda x: get_dr_session(dataset, x)
    )

    # Awake and Awake_Rest have several runs per session, each with its
    # own FD / tSNR: match on the run too, otherwise every run would get
    # the metadata of run 1
    match_keys = ["match_id", "match_session"]

    if "Run" in metadata.columns:

        metadata["match_run"] = pd.to_numeric(metadata["Run"], errors="coerce")

        run_in_name = dr["subject"].str.extract(r"_run-(\d+)", expand=False)

        if run_in_name.notna().all():
            dr["match_run"] = run_in_name.astype(int)
        else:
            # Awake_Rest: DR file has no run in the name,
            # rows are in run order within each subject/session
            dr["match_run"] = dr.groupby(match_keys).cumcount() + 1

        match_keys = match_keys + ["match_run"]

    # Sexdifference: match only on subject ID
    if dataset == "Sexdifference":

        metadata_unique = metadata.drop_duplicates(
            subset=["match_id"]
        )

        matched = dr.merge(
            metadata_unique,
            on="match_id",
            how="left"
        )

    else:

        metadata_unique = metadata.drop_duplicates(
            subset=match_keys
        )

        matched = dr.merge(
            metadata_unique,
            on=match_keys,
            how="left"
        )

    # Use the session from the DR scan
    if "match_session_x" in matched.columns:
        matched["Session"] = matched["match_session_x"]
    else:
        matched["Session"] = matched["match_session"]

    # Fill existing Scan_id with DR subject
    matched["Scan_id"] = matched["subject"]

    # Remove original DR subject column
    matched = matched.drop(columns="subject")

    # Dataset name
    matched["Dataset_name"] = dataset

    # Remove temporary matching columns
    columns_to_remove = [
        "match_id",
        "match_session",
        "match_session_x",
        "match_session_y",
        "match_run",
        "Run"
    ]

    for column in columns_to_remove:
        if column in matched.columns:
            matched = matched.drop(columns=column)

    all_datasets.append(matched)


# Combine all datasets
Combined_dataset = pd.concat(
    all_datasets,
    ignore_index=True
)

# Rearrange columns
first_columns = [
    "Scan_id",
    "Participant_Id",
    "Session",
    "Dataset_name",
    "Sex",
    "Age(week)",
    "Rodent.strain",
    "MRI",
    "Coil",
    "fMRI.sequence",
    "MRI T.R.",
    "Anesthesia",
    "Ventilation",
    "Anesthesia level",
    "FD average",
    "FD maximum",
    "tSNR",
]

components = [f"comp{i}" for i in range(1, 19)]

Combined_dataset = Combined_dataset[
    first_columns + components
]

# Save
Combined_dataset.to_csv(
    out_dir / "Combined_dataset.csv",
    index=False
)

print(Combined_dataset.shape)
```

    Skipping dataset Propofol (in excluded_datasets)
    (2301, 35)

``` python
# ============================================================
# CLEAN COMBINED DATASET
# ============================================================

# ------------------------------------------------------------
# Sex
# ------------------------------------------------------------

Combined_dataset["Sex"] = (
    Combined_dataset["Sex"]
    .astype("string")
    .str.strip()
    .str.upper()
)

Combined_dataset["Sex"] = Combined_dataset["Sex"].replace({
    "FEMALE": "F",
    "MALE": "M"
})


# ------------------------------------------------------------
# Rodent strain
# ------------------------------------------------------------

Combined_dataset["Rodent.strain"] = (
    Combined_dataset["Rodent.strain"]
    .astype("string")
    .str.strip()
)

Combined_dataset.loc[
    Combined_dataset["Rodent.strain"].str.contains(
        "C57",
        case=False,
        na=False
    ),
    "Rodent.strain"
] = "C57BL/6"

Combined_dataset["Rodent.strain"] = Combined_dataset["Rodent.strain"].replace({
    "missing": "C57BL/6",
    "na": "C57BL/6",
    "n/a": "C57BL/6"
})

# ------------------------------------------------------------
# MRI field strength
# ------------------------------------------------------------

Combined_dataset["MRI"] = pd.to_numeric(
    Combined_dataset["MRI"],
    errors="coerce"
)


# ------------------------------------------------------------
# Coil
# Missing / N.A. = no cryocoil = RT
# ------------------------------------------------------------

Combined_dataset["Coil"] = (
    Combined_dataset["Coil"]
    .astype("string")
    .str.strip()
    .str.lower()
)

Combined_dataset["Coil"] = Combined_dataset["Coil"].replace({
    "y": "Cryo",
    "yes": "Cryo",
    "cryo": "Cryo",
    "n": "RT",
    "no": "RT",
    "rt": "RT",
    "missing": "RT",
    "na": "RT",
    "n/a": "RT"
})

Combined_dataset["Coil"] = (
    Combined_dataset["Coil"]
    .fillna("RT")
)


# ------------------------------------------------------------
# fMRI sequence
# ------------------------------------------------------------

Combined_dataset["fMRI.sequence"] = (
    Combined_dataset["fMRI.sequence"]
    .astype("string")
    .str.strip()
    .str.upper()
)

Combined_dataset["fMRI.sequence"] = (
    Combined_dataset["fMRI.sequence"].replace({
        "EPI": "GE-EPI",
        "GE EPI": "GE-EPI",
        "SE EPI": "SE-EPI",
        "ME EPI": "ME-EPI"
    })
)

# ------------------------------------------------------------
# Anesthesia
# Missing / N.A. = Awake
# ------------------------------------------------------------

Combined_dataset["Anesthesia"] = (
    Combined_dataset["Anesthesia"]
    .astype("string")
    .str.strip()
    .str.lower()
)

Combined_dataset["Anesthesia"] = Combined_dataset["Anesthesia"].replace({
    "awake": "Awake",
    "iso": "Iso",
    "dex": "Dex",
    "dexiso": "Dexiso",
    "med": "Med",
    "mediso": "Mediso",
    "hal": "Hal",
    "propofol": "Propofol",
    "missing": "Awake",
    "na": "Awake",
    "n/a": "Awake"
})

Combined_dataset["Anesthesia"] = (
    Combined_dataset["Anesthesia"]
    .fillna("Awake")
)


# ------------------------------------------------------------
# Ventilation
# Missing / N.A. = No
# ------------------------------------------------------------

Combined_dataset["Ventilation"] = (
    Combined_dataset["Ventilation"]
    .astype("string")
    .str.strip()
    .str.lower()
)

Combined_dataset["Ventilation"] = Combined_dataset["Ventilation"].replace({
    "y": "Yes",
    "yes": "Yes",
    "n": "No",
    "no": "No",
    "missing": "No",
    "na": "No",
    "n/a": "No"
})

Combined_dataset["Ventilation"] = (
    Combined_dataset["Ventilation"]
    .fillna("No")
)


# ------------------------------------------------------------
# Anesthesia level (dose): Low / High; awake scans = None
# Missing for an anesthetized scan stays missing
# ------------------------------------------------------------

Combined_dataset["Anesthesia level"] = (
    Combined_dataset["Anesthesia level"]
    .astype("string")
    .str.strip()
    .str.lower()
    .replace({"l": "low", "h": "high", "": pd.NA})
    .replace({"low": "Low", "high": "High"})
)

Combined_dataset.loc[
    Combined_dataset["Anesthesia"] == "Awake", "Anesthesia level"
] = "Awake"


# ------------------------------------------------------------
# Numeric columns
# ------------------------------------------------------------

numeric_columns = [
    "Age(week)",
    "MRI",
    "MRI T.R.",
    "FD average",
    "FD maximum",
    "tSNR"
]

for col in numeric_columns:
    Combined_dataset[col] = pd.to_numeric(
        Combined_dataset[col],
        errors="coerce"
    )


# ============================================================
# CHECK CLEANED VALUES
# ============================================================

categorical_columns = [
    "Sex",
    "Rodent.strain",
    "Coil",
    "fMRI.sequence",
    "Anesthesia",
    "Ventilation"
]

for col in categorical_columns:
    print("\n" + col)
    print(
        Combined_dataset[col]
        .value_counts(dropna=False)
    )


print("\nMissing required metadata:")

required_columns = [
    "Age(week)",
    "Sex",
    "MRI",
    "MRI T.R."
]

for col in required_columns:
    print("\n" + col)

    print(
        Combined_dataset[
            Combined_dataset[col].isna()
        ]
        .groupby("Dataset_name")
        .size()
    )


# ============================================================
# SAVE CLEAN COMBINED DATASET
# ============================================================

Combined_dataset.to_csv(
    out_dir / "Combined_dataset.csv",
    index=False
)

print("\nSaved cleaned Combined_dataset.csv")
print(Combined_dataset.shape)
```

    Sex
    Sex
    M       1140
    F       1011
    <NA>     150
    Name: count, dtype: int64[pyarrow]

    Rodent.strain
    Rodent.strain
    C57BL/6        1716
    F1 C6/129P      476
    129S2/SvPas     105
    ICR               4
    Name: count, dtype: int64[pyarrow]

    Coil
    Coil
    RT      1978
    Cryo     323
    Name: count, dtype: int64[pyarrow]

    fMRI.sequence
    fMRI.sequence
    GE-EPI    1217
    SE-EPI    1045
    ME-EPI      39
    Name: count, dtype: int64[pyarrow]

    Anesthesia
    Anesthesia
    Awake     1621
    Mediso     414
    Iso        149
    Med         80
    Hal         37
    Name: count, dtype: int64[pyarrow]

    Ventilation
    Ventilation
    No     1928
    Yes     373
    Name: count, dtype: int64[pyarrow]

    Missing required metadata:

    Age(week)
    Dataset_name
    Awake    224
    dtype: int64

    Sex
    Dataset_name
    Multicenter    150
    dtype: int64

    MRI
    Series([], dtype: int64)

    MRI T.R.
    Series([], dtype: int64)

    Saved cleaned Combined_dataset.csv
    (2301, 35)

``` python
# ============================================================
# REMOVE EXCLUDED SCANS
# ============================================================

exclude = pd.read_csv(
    root / "data" / "Exclude.tsv",
    sep="\t",
    dtype=str
)

exclude.columns = exclude.columns.str.strip()


def exclusion_id(dataset, subject):

    subject = str(subject).strip().lower()

    # Keep Longitudinal age information as part of the ID
    if dataset == "Longitudinal":
        subject = subject.replace("_ses-", "-")

    # Remove session, run, task and anatomical suffixes
    subject = re.sub(r"_ses-?\d+.*$", "", subject)
    subject = re.sub(r"_run-\d+.*$", "", subject)
    subject = re.sub(r"_task-.*$", "", subject)
    subject = re.sub(r"_t2w$", "", subject)

    subject = subject.removeprefix("sub-")

    if subject.isdigit():
        subject = subject.lstrip("0") or "0"

    return subject


# Awake_Rest: add missing run numbers
# Assumes rows within each subject/session are ordered run 1, 2, 3
awake_rest = (
    (Combined_dataset["Dataset_name"] == "Awake_Rest")
    & ~Combined_dataset["Scan_id"].str.contains(
        r"_run-\d+",
        na=False
    )
)

ids = (
    Combined_dataset.loc[awake_rest, "Scan_id"]
    .str.strip()
)

runs = ids.groupby(ids, sort=False).cumcount() + 1

Combined_dataset.loc[awake_rest, "Scan_id"] = (
    ids + "_run-" + runs.astype(str)
)


# Get scan IDs, sessions and runs
scan_ids = Combined_dataset["Scan_id"].astype(str).str.strip()

scan_sessions = pd.to_numeric(
    Combined_dataset["Session"],
    errors="coerce"
)

scan_runs = pd.to_numeric(
    scan_ids.str.extract(r"_run-(\d+)", expand=False),
    errors="coerce"
)

remove = pd.Series(False, index=Combined_dataset.index)


# Match exclusions
for dataset in exclude["Dataset"].dropna().unique():

    dataset_rows = Combined_dataset["Dataset_name"] == dataset

    subject_ids = scan_ids.apply(
        lambda x: exclusion_id(dataset, x)
    )

    for subject in exclude.loc[
        exclude["Dataset"] == dataset, "Subject"
    ].dropna():

        subject = subject.strip()
        subject_id = exclusion_id(dataset, subject)

        match = dataset_rows & (subject_ids == subject_id)

        session = re.search(r"_ses-?(\d+)", subject, re.IGNORECASE)
        run = re.search(r"_run-(\d+)", subject, re.IGNORECASE)

        if session:
            match = match & (
                scan_sessions == int(session.group(1))
            )

        if run:
            match = match & (
                scan_runs == int(run.group(1))
            )

        remove = remove | match


# Show removals
print("\nRemoved scans per dataset:")
print(
    Combined_dataset.loc[remove]
    .groupby("Dataset_name")
    .size()
)

print("\nTotal scans removed:", remove.sum())


# Remove excluded scans
Combined_dataset = (
    Combined_dataset.loc[~remove]
    .reset_index(drop=True)
)


# Save
Combined_dataset.to_csv(
    out_dir / "Combined_dataset.csv",
    index=False
)

print("\nSaved Combined_dataset.csv")
print("Remaining shape:", Combined_dataset.shape)
```

    Removed scans per dataset:
    Dataset_name
    Awake             114
    Awake_Rest          5
    Env_Enrichment      1
    Longitudinal       16
    Multicenter        14
    NDUFS4              3
    dtype: int64

    Total scans removed: 153

    Saved Combined_dataset.csv
    Remaining shape: (2148, 35)
