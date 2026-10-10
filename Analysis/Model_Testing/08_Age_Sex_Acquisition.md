# Model Testing - Age + Sex + Acquisition


``` python
from pathlib import Path
import numpy as np
import pandas as pd

from pcntoolkit import (
    BLR,
    NormativeModel,
    NormData,
    plot_qq
)

import pcntoolkit.util.output


# ============================================================
# FIND PROJECT ROOT
# ============================================================

here = Path.cwd()

candidates = [
    here,
    here.parent,
    here.parent.parent
]

current_dir = next(
    p for p in candidates
    if (
        p
        / "results"
        / "combined_dataset"
        / "Model_dataframe.csv"
    ).exists()
)


# ============================================================
# LOAD MODEL DATAFRAME
# ============================================================

model_df = pd.read_csv(
    current_dir
    / "results"
    / "combined_dataset"
    / "Model_dataframe.csv"
)

pcntoolkit.util.output.Output.set_show_messages(False)

np.random.seed(42)


# ============================================================
# COLUMN GROUPS
# ============================================================

age_cols = [
    col for col in [
        "Age(week)",
        "Age(week)_Missing"
    ]
    if col in model_df.columns
]

sex_cols = [
    col for col in [
        "Sex_M",
        "Sex_Missing"
    ]
    if col in model_df.columns
]

strain_cols = [
    col for col in model_df.columns
    if col.startswith("Rodent.strain_")
]

mri_cols = [
    col for col in [
        "MRI",
        "MRI_Missing"
    ]
    if col in model_df.columns
]

tr_cols = [
    col for col in [
        "MRI T.R.",
        "MRI T.R._Missing"
    ]
    if col in model_df.columns
]

coil_cols = [
    col for col in model_df.columns
    if col.startswith("Coil_")
]

sequence_cols = [
    col for col in model_df.columns
    if col.startswith("fMRI.sequence_")
]

anesthesia_cols = [
    col for col in model_df.columns
    if col.startswith("Anesthesia_")
]

ventilation_cols = [
    col for col in model_df.columns
    if col.startswith("Ventilation_")
]

fd_cols = [
    col for col in [
        "FD average",
        "FD average_Missing",
        "FD maximum",
        "FD maximum_Missing"
    ]
    if col in model_df.columns
]


# ============================================================
# MODEL SPECIFICATION
# ============================================================

model_name = "08_Age_Sex_Acquisition"

response_vars = [f"comp{i}" for i in range(1, 19)]

covariates = age_cols + sex_cols + mri_cols + tr_cols + coil_cols + sequence_cols + anesthesia_cols + ventilation_cols


# ============================================================
# CHECK MODEL SETUP
# ============================================================

batch_effects = []

print("Responses:")
print(response_vars)

print("\nCovariates:")
print(covariates)

print("\nBatch effects:")
print(batch_effects)

print("\nRows:")
print(len(model_df))

print("\nMissing covariate values:")
print(model_df[covariates].isna().sum().sum())

print("\nMissing response values:")
print(model_df[response_vars].isna().sum().sum())


# ============================================================
# CREATE NORMDATA
# ============================================================

norm_data = NormData.from_dataframe(
    name=model_name,
    dataframe=model_df,
    covariates=covariates,
    batch_effects=batch_effects,
    response_vars=response_vars,
    remove_Nan=False
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train, test = norm_data.train_test_split()


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

model_dir = (
    current_dir
    / "results"
    / "normative_model"
    / "Model_Testing"
    / model_name
)

model_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# BLR MODEL
# ============================================================

model = NormativeModel(
    BLR(
        heteroskedastic=True
    ),
    inscaler="standardize",
    outscaler="standardize",
    savemodel=True,
    evaluate_model=True,
    saveresults=True,
    saveplots=True,
    save_dir=str(model_dir)
)


# ============================================================
# FIT MODEL
# ============================================================

model.fit_predict(
    train,
    test
)


# ============================================================
# QQ PLOTS
# ============================================================

qq_dir = (
    model_dir
    / "plots"
    / "qq"
)

qq_dir.mkdir(
    parents=True,
    exist_ok=True
)

plot_qq(
    test,
    plot_id_line=True,
    save_dir=str(qq_dir)
)


# ============================================================
# SAVE STATISTICS
# ============================================================

train_stats = train.get_statistics_df()
test_stats = test.get_statistics_df()

train_stats.to_csv(
    model_dir / "Train_statistics.csv",
    index=False
)

test_stats.to_csv(
    model_dir / "Test_statistics.csv",
    index=False
)

print("\nTrain statistics:")
display(train_stats)

print("\nTest statistics:")
display(test_stats)

print("\nFinished:")
print(model_name)
print(model_dir)
```

    Responses:
    ['comp1', 'comp2', 'comp3', 'comp4', 'comp5', 'comp6', 'comp7', 'comp8', 'comp9', 'comp10', 'comp11', 'comp12', 'comp13', 'comp14', 'comp15', 'comp16', 'comp17', 'comp18']

    Covariates:
    ['Age(week)', 'Age(week)_Missing', 'Sex_M', 'Sex_Missing', 'MRI', 'MRI_Missing', 'MRI T.R.', 'MRI T.R._Missing', 'Coil_Cryo', 'fMRI.sequence_ME-EPI', 'fMRI.sequence_SE-EPI', 'fMRI.sequence_Missing', 'Anesthesia_Dex', 'Anesthesia_Dexiso', 'Anesthesia_Hal', 'Anesthesia_Iso', 'Anesthesia_Med', 'Anesthesia_Mediso', 'Anesthesia_Propofol', 'Ventilation_Yes']

    Batch effects:
    []

    Rows:
    2020

    Missing covariate values:
    0

    Missing response values:
    0

    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.29587122877626e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.0692505888333915e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.021753145635003e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.67098292850794e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:03:58 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.0451256958674575e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.267896286912639e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.378382402898118e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.109333538909041e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.688557180255335e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.0727611043012087e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.0280938022248015e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.6664713669658152e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.439806328070732e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.05060981073577e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.581916947308785e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.727197306037593e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.652027857126284e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.0371498730782397e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.1255537424668952e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.210639302907571e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.0277849178723986e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/scipy/optimize/_numdiff.py:687: RuntimeWarning: overflow encountered in divide
      df_dx = [delf / delx for delf, delx in zip(df, dx)]
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:03:59 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.1108077416472007e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:03:59 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.109272998775333e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.492229384100894e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.867588709158092e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.141446622747662e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.8502359151408235e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.879616148629273e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.934401410422818e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:01 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.626473278753641e-23.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:01 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.290983416847483e-23.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:03 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.8206519483396362e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:05 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:05 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.032124344645038e-23.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.004115353086411e-23.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.72116338788577e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.734033712032274e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.849986413595935e-23.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.206397807759912e-23.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.6074420707854608e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.5090757242676583e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.554016275184348e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:06 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.282236260288374e-37.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:06 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.811279949056961e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.804117132972966e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.428609839997238e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.685967424190354e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.857234574616112e-37.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.083614898014294e-31.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:10 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:10 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.4311697354771857e-25.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.9314296379833107e-31.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.195895010287291e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.1046524698194493e-32.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.831730681312871e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.0863576437413619e-31.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.1864432307020398e-30.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.5916423559247346e-31.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.872657597323333e-32.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.2837704309027456e-29.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.849164585501168e-32.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:13 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.0692461663697658e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:13 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.025398423709915e-24.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.6320394233996365e-23.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.2153794469612005e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.522945596148885e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.516469425728372e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.1186289586976933e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.8037085878607957e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.6061155622566266e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.0585488828018657e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.4997562764097525e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.648064505266034e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:15 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.258063526022474e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:15 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.334712179456937e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.381129603877223e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.16831045937323e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.2874238870134448e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.0331022322779292e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.7229906897408152e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.9212150758381546e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.0949651372974954e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.825569474345168e-35.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.226973578412322e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.47247966219053e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.058252643292565e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:16 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.736818540791051e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:16 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.827437977063485e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.318624543456142e-24.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.218556086541799e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.563847386213581e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.8475044542051765e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.1634544605730824e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.780544630185892e-23.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.5015513672675644e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.266382541028155e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.821126022778487e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.984455305859157e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:21 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.322738883137918e-23.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:21 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.909786455991528e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.4092447643711246e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.7749245924111267e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.222113913933542e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.5646488466300533e-37.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4011915 - 2026-10-10 12:04:22 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.556131972896186e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.555954910341371e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.047819251391431e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.3264553928942246e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.2909523215791866e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.3817076588962484e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.181038658892322e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.7475382112316367e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.0442365252543695e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.682662916280195e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.676326716320041e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.83594529472095e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.1952476120366853e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.4593986755460144e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.55450819438763e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.4336526657060027e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.784911435030191e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.7032580968606615e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.770406638263375e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.8344917328936724e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.981254573108124e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)


    Train statistics:

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

| statistic | EXPV | MACE | MAPE | MSLL | NLL | R2 | RMSE | Rho | Rho_p | SMSE | ShapiroW |
|----|----|----|----|----|----|----|----|----|----|----|----|
| response_vars |  |  |  |  |  |  |  |  |  |  |  |
| comp1 | 0.679313 | 0.016337 | 0.707291 | 0.424410 | 0.523320 | 0.679312 | 0.151273 | 0.835174 | 0.000000e+00 | 0.320688 | 0.983222 |
| comp10 | 0.597273 | 0.024134 | 0.645827 | 1.411532 | 0.635012 | 0.596890 | 0.070670 | 0.833024 | 0.000000e+00 | 0.403110 | 0.969537 |
| comp11 | 0.620895 | 0.036386 | 0.556085 | 1.067368 | 0.661277 | 0.620795 | 0.099274 | 0.828689 | 0.000000e+00 | 0.379205 | 0.938308 |
| comp12 | 0.613895 | 0.039356 | 0.513014 | 1.159062 | 0.690988 | 0.613092 | 0.094251 | 0.818249 | 0.000000e+00 | 0.386908 | 0.914172 |
| comp13 | 0.704639 | 0.025495 | 0.601276 | 1.398777 | 0.595270 | 0.702874 | 0.059057 | 0.813389 | 0.000000e+00 | 0.297126 | 0.982322 |
| comp14 | 0.632108 | 0.030322 | 0.596640 | 1.628206 | 0.683545 | 0.632081 | 0.057066 | 0.814410 | 0.000000e+00 | 0.367919 | 0.958834 |
| comp15 | 0.701856 | 0.024505 | 0.556071 | 1.046406 | 0.551436 | 0.701853 | 0.080541 | 0.850824 | 0.000000e+00 | 0.298147 | 0.979183 |
| comp16 | 0.651813 | 0.027673 | 0.701561 | 0.773593 | 0.505607 | 0.651124 | 0.109324 | 0.832370 | 0.000000e+00 | 0.348876 | 0.962124 |
| comp17 | 0.712985 | 0.022277 | 2.884971 | 1.147101 | 0.535045 | 0.712736 | 0.070322 | 0.853621 | 0.000000e+00 | 0.287264 | 0.969478 |
| comp18 | 0.563952 | 0.032178 | 0.734696 | 1.180455 | 0.593226 | 0.561413 | 0.089076 | 0.822054 | 0.000000e+00 | 0.438587 | 0.959289 |
| comp2 | 0.629859 | 0.028960 | 0.467262 | 1.130375 | 0.692172 | 0.629004 | 0.095091 | 0.850130 | 0.000000e+00 | 0.370996 | 0.960842 |
| comp3 | 0.598714 | 0.027351 | 0.539334 | 1.721093 | 0.834958 | 0.598529 | 0.063204 | 0.745082 | 3.434531e-286 | 0.401471 | 0.967745 |
| comp4 | 0.565773 | 0.026856 | 0.680370 | 0.844741 | 0.600870 | 0.564649 | 0.125104 | 0.840351 | 0.000000e+00 | 0.435351 | 0.965925 |
| comp5 | 0.704383 | 0.026856 | 1.324021 | 1.351860 | 0.620574 | 0.704014 | 0.063358 | 0.853557 | 0.000000e+00 | 0.295986 | 0.971914 |
| comp6 | 0.652847 | 0.029579 | 0.521415 | 1.556121 | 0.738190 | 0.652293 | 0.062972 | 0.810421 | 0.000000e+00 | 0.347707 | 0.956725 |
| comp7 | 0.593595 | 0.036460 | 0.749013 | 0.744885 | 0.620665 | 0.587403 | 0.137271 | 0.823255 | 0.000000e+00 | 0.412597 | 0.930370 |
| comp8 | 0.588940 | 0.035149 | 0.516812 | 1.395288 | 0.679166 | 0.588575 | 0.075840 | 0.769412 | 0.000000e+00 | 0.411425 | 0.938542 |
| comp9 | 0.635548 | 0.030446 | 0.529736 | 1.041944 | 0.666655 | 0.634308 | 0.100539 | 0.832260 | 0.000000e+00 | 0.365692 | 0.964643 |

</div>


    Test statistics:

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

| statistic | EXPV | MACE | MAPE | MSLL | NLL | R2 | RMSE | Rho | Rho_p | SMSE | ShapiroW |
|----|----|----|----|----|----|----|----|----|----|----|----|
| response_vars |  |  |  |  |  |  |  |  |  |  |  |
| comp1 | 0.704726 | 0.021782 | 1.333791e+12 | 0.411214 | 0.472000 | 0.704341 | 0.139816 | 0.850937 | 1.830307e-114 | 0.295659 | 0.961659 |
| comp10 | 0.624953 | 0.027228 | 6.290153e-01 | 1.477149 | 0.699367 | 0.622356 | 0.068315 | 0.851594 | 8.080530e-115 | 0.377644 | 0.922347 |
| comp11 | 0.651350 | 0.042574 | 4.846230e-01 | 1.142388 | 0.749126 | 0.651240 | 0.096435 | 0.837451 | 1.552802e-107 | 0.348760 | 0.863044 |
| comp12 | 0.656539 | 0.037129 | 5.044939e-01 | 1.280550 | 0.763474 | 0.655892 | 0.084634 | 0.822455 | 1.537267e-100 | 0.344108 | 0.812522 |
| comp13 | 0.742059 | 0.019802 | 4.700065e-01 | 1.331958 | 0.500195 | 0.741999 | 0.053499 | 0.827774 | 6.072076e-103 | 0.258001 | 0.986618 |
| comp14 | 0.689148 | 0.041089 | 5.682422e-01 | 1.684072 | 0.701549 | 0.687987 | 0.050599 | 0.837307 | 1.827408e-107 | 0.312013 | 0.898652 |
| comp15 | 0.731028 | 0.029208 | 1.300200e+12 | 1.057830 | 0.510758 | 0.730954 | 0.072625 | 0.845933 | 8.147597e-112 | 0.269046 | 0.959373 |
| comp16 | 0.697524 | 0.022772 | 6.393629e-01 | 0.848148 | 0.529797 | 0.697432 | 0.096809 | 0.843221 | 2.027190e-110 | 0.302568 | 0.928683 |
| comp17 | 0.752993 | 0.028218 | 9.604576e-01 | 1.149097 | 0.498278 | 0.752463 | 0.062797 | 0.860876 | 5.027444e-120 | 0.247537 | 0.953454 |
| comp18 | 0.627073 | 0.047525 | 6.304256e-01 | 1.341338 | 0.697623 | 0.625049 | 0.077837 | 0.834946 | 2.561588e-106 | 0.374951 | 0.859649 |
| comp2 | 0.655594 | 0.032178 | 5.227955e-01 | 1.268846 | 0.804081 | 0.653711 | 0.089462 | 0.830869 | 2.220583e-104 | 0.346289 | 0.895777 |
| comp3 | 0.615909 | 0.034158 | 4.649692e-01 | 1.767954 | 0.885488 | 0.615179 | 0.062107 | 0.738095 | 1.076925e-70 | 0.384821 | 0.921242 |
| comp4 | 0.576831 | 0.031683 | 7.099069e-01 | 1.121440 | 0.894498 | 0.572770 | 0.126047 | 0.832390 | 4.263447e-105 | 0.427230 | 0.798721 |
| comp5 | 0.718151 | 0.032871 | 4.626789e-01 | 1.383240 | 0.645528 | 0.718066 | 0.061440 | 0.851858 | 5.811767e-115 | 0.281934 | 0.885748 |
| comp6 | 0.657618 | 0.028218 | 6.097567e-01 | 1.604741 | 0.799766 | 0.657607 | 0.063304 | 0.797642 | 2.646618e-90 | 0.342393 | 0.901388 |
| comp7 | 0.638465 | 0.036139 | 5.687811e-01 | 0.862655 | 0.730085 | 0.638154 | 0.127482 | 0.834139 | 6.254909e-106 | 0.361846 | 0.859860 |
| comp8 | 0.631650 | 0.032178 | 7.452224e-01 | 1.459446 | 0.729556 | 0.630567 | 0.070883 | 0.772603 | 2.485325e-81 | 0.369433 | 0.881840 |
| comp9 | 0.700714 | 0.040594 | 6.741048e-01 | 1.100556 | 0.665095 | 0.699108 | 0.085872 | 0.827745 | 6.259762e-103 | 0.300892 | 0.919313 |

</div>


    Finished:
    08_Age_Sex_Acquisition
    /home/traaffneu/dansch/Documents/digital_rodent/digitalrodent/results/normative_model/Model_Testing/08_Age_Sex_Acquisition
