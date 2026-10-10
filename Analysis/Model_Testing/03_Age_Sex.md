# Model Testing - Age + Sex


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

model_name = "03_Age_Sex"

response_vars = [f"comp{i}" for i in range(1, 19)]

covariates = age_cols + sex_cols


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
    ['Age(week)', 'Age(week)_Missing', 'Sex_M', 'Sex_Missing']

    Batch effects:
    []

    Rows:
    2020

    Missing covariate values:
    0

    Missing response values:
    0

    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.901237137085496e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:54 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.888937237106057e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.598931721661133e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.194021262160357e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.016171979351757e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.5379509706283017e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:54 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.6236825575122716e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:54 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.099789438783885e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.253819177124223e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.049166831038045e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.49266330127728e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/scipy/optimize/_numdiff.py:687: RuntimeWarning: overflow encountered in divide
      df_dx = [delf / delx for delf, delx in zip(df, dx)]
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.1634745492975198e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:54 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.1289699807667585e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.045355153074208e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.1489855618779247e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:54 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.876162775469817e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.494101173370663e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:54 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.46916586204203e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.134087571378479e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.303200810630474e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.1318369560598774e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.7837547474686995e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/scipy/optimize/_numdiff.py:687: RuntimeWarning: overflow encountered in divide
      df_dx = [delf / delx for delf, delx in zip(df, dx)]
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.4043258183967767e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:54 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.818190751122715e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.128275946697949e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.5706839085863794e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:54 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.741293394388879e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.839520073399999e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.037877023294131e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:55 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.4539418553093595e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.016473845733149e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.5740799474644664e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.8685569437811528e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/scipy/optimize/_numdiff.py:687: RuntimeWarning: overflow encountered in divide
      df_dx = [delf / delx for delf, delx in zip(df, dx)]
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.781797232935279e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:55 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.991982156574467e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.220161518147032e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.577383174106605e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.618459443036949e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.100990658569585e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/scipy/optimize/_numdiff.py:687: RuntimeWarning: overflow encountered in divide
      df_dx = [delf / delx for delf, delx in zip(df, dx)]
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.097710719618444e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:55 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.537171591128321e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.105984905828453e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.368914881435389e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:55 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.9952741966215077e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.375794213931526e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:55 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.8651739225600992e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.3324561484352901e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.253911001511913e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:55 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.1615277955475765e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.352161258652272e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:56 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.6612149973633066e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:56 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.627296292268013e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.6827632060082266e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.688077190697632e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:56 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.082207973988209e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4006688 - 2026-10-10 11:59:56 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.1124718524198865e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.372464139376449e-19.
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
| comp1 | 0.143586 | 0.049381 | 1.604067 | 1.126545 | 1.225455 | 0.139512 | 0.247794 | 0.319443 | 1.169847e-39 | 0.860488 | 0.927709 |
| comp10 | 0.126096 | 0.035149 | 1.107134 | 2.033228 | 1.256707 | 0.126048 | 0.104056 | 0.246230 | 9.594004e-24 | 0.873952 | 0.952622 |
| comp11 | 0.114667 | 0.026114 | 1.166369 | 1.703164 | 1.297074 | 0.114666 | 0.151689 | 0.201276 | 3.117640e-16 | 0.885334 | 0.954614 |
| comp12 | 0.111528 | 0.024629 | 1.088209 | 1.790140 | 1.322065 | 0.111525 | 0.142825 | 0.206141 | 5.742430e-17 | 0.888475 | 0.946974 |
| comp13 | 0.152290 | 0.030520 | 1.053586 | 2.079513 | 1.276005 | 0.152225 | 0.099757 | 0.309176 | 3.910609e-37 | 0.847775 | 0.955739 |
| comp14 | 0.116824 | 0.027970 | 1.141055 | 2.239092 | 1.294431 | 0.116735 | 0.088419 | 0.258580 | 4.239582e-26 | 0.883265 | 0.965938 |
| comp15 | 0.133035 | 0.033490 | 1.169000 | 1.758168 | 1.263198 | 0.132976 | 0.137346 | 0.251485 | 9.904479e-25 | 0.867024 | 0.954331 |
| comp16 | 0.129307 | 0.040594 | 1.618232 | 1.539398 | 1.271413 | 0.128454 | 0.172792 | 0.266050 | 1.381220e-27 | 0.871546 | 0.930960 |
| comp17 | 0.124528 | 0.034975 | 2.080401 | 1.896326 | 1.284270 | 0.124527 | 0.122764 | 0.208148 | 2.821542e-17 | 0.875473 | 0.946896 |
| comp18 | 0.106528 | 0.030569 | 1.541134 | 1.849357 | 1.262128 | 0.106418 | 0.127145 | 0.201466 | 2.920983e-16 | 0.893582 | 0.944159 |
| comp2 | 0.141357 | 0.029208 | 0.959459 | 1.674670 | 1.236467 | 0.141317 | 0.144667 | 0.286461 | 6.746035e-32 | 0.858683 | 0.962470 |
| comp3 | 0.151200 | 0.020371 | 0.911727 | 2.213652 | 1.327517 | 0.151199 | 0.091901 | 0.277827 | 4.981231e-30 | 0.848801 | 0.979134 |
| comp4 | 0.121455 | 0.041337 | 1.356938 | 1.477270 | 1.233398 | 0.120826 | 0.177782 | 0.248297 | 3.952701e-24 | 0.879174 | 0.932553 |
| comp5 | 0.142542 | 0.029406 | 1.469866 | 2.000434 | 1.269148 | 0.142528 | 0.107840 | 0.248191 | 4.137197e-24 | 0.857472 | 0.964570 |
| comp6 | 0.138391 | 0.030693 | 0.976305 | 2.116175 | 1.298244 | 0.138391 | 0.099128 | 0.276560 | 9.250047e-30 | 0.861609 | 0.965030 |
| comp7 | 0.121011 | 0.047896 | 1.474834 | 1.389457 | 1.265237 | 0.120812 | 0.200381 | 0.145065 | 4.683504e-09 | 0.879188 | 0.925030 |
| comp8 | 0.110643 | 0.027550 | 1.109453 | 2.059446 | 1.343325 | 0.110642 | 0.111505 | 0.188356 | 2.269276e-14 | 0.889358 | 0.948606 |
| comp9 | 0.123716 | 0.025000 | 1.178973 | 1.636231 | 1.260941 | 0.123709 | 0.155633 | 0.229239 | 1.031812e-20 | 0.876291 | 0.961881 |

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
| comp1 | 0.136267 | 0.050000 | 1.968200e+12 | 1.115072 | 1.175858 | 0.136103 | 0.238997 | 0.283316 | 6.786614e-09 | 0.863897 | 0.925473 |
| comp10 | 0.104080 | 0.034158 | 1.222301e+00 | 2.022407 | 1.244625 | 0.103888 | 0.105234 | 0.179351 | 2.910142e-04 | 0.896112 | 0.951864 |
| comp11 | 0.103885 | 0.034356 | 1.307998e+00 | 1.678368 | 1.285106 | 0.102966 | 0.154659 | 0.152625 | 2.096146e-03 | 0.897034 | 0.950603 |
| comp12 | 0.090106 | 0.033366 | 1.258779e+00 | 1.796933 | 1.279857 | 0.085061 | 0.138005 | 0.137711 | 5.561080e-03 | 0.914939 | 0.950596 |
| comp13 | 0.125852 | 0.040792 | 1.088502e+00 | 2.085484 | 1.253722 | 0.124796 | 0.098534 | 0.244998 | 6.186241e-07 | 0.875204 | 0.944142 |
| comp14 | 0.099600 | 0.031386 | 1.245094e+00 | 2.249615 | 1.267093 | 0.099267 | 0.085972 | 0.205516 | 3.147167e-05 | 0.900733 | 0.962551 |
| comp15 | 0.108450 | 0.038812 | 1.504957e+12 | 1.790178 | 1.243106 | 0.104886 | 0.132468 | 0.189700 | 1.250575e-04 | 0.895114 | 0.931398 |
| comp16 | 0.116365 | 0.039604 | 1.806506e+00 | 1.534289 | 1.215937 | 0.115881 | 0.165485 | 0.238510 | 1.238982e-06 | 0.884119 | 0.936417 |
| comp17 | 0.102449 | 0.042772 | 1.493906e+00 | 1.913385 | 1.262566 | 0.097394 | 0.119913 | 0.146296 | 3.205918e-03 | 0.902606 | 0.939478 |
| comp18 | 0.092704 | 0.035842 | 1.774512e+00 | 1.857637 | 1.213922 | 0.089928 | 0.121266 | 0.162360 | 1.056489e-03 | 0.910072 | 0.937493 |
| comp2 | 0.118412 | 0.035149 | 1.130967e+00 | 1.721237 | 1.256472 | 0.117503 | 0.142815 | 0.233714 | 2.044633e-06 | 0.882497 | 0.954050 |
| comp3 | 0.119465 | 0.020495 | 9.000963e-01 | 2.229146 | 1.346681 | 0.118095 | 0.094021 | 0.217818 | 9.977198e-06 | 0.881905 | 0.973485 |
| comp4 | 0.121421 | 0.047030 | 1.534153e+00 | 1.453612 | 1.226671 | 0.121390 | 0.180760 | 0.217625 | 1.016371e-05 | 0.878610 | 0.931979 |
| comp5 | 0.116051 | 0.041782 | 1.224709e+00 | 2.020629 | 1.282917 | 0.113420 | 0.108953 | 0.200626 | 4.877886e-05 | 0.886580 | 0.948153 |
| comp6 | 0.128125 | 0.035842 | 1.164242e+00 | 2.121609 | 1.316633 | 0.126360 | 0.101119 | 0.256874 | 1.647987e-07 | 0.873640 | 0.954778 |
| comp7 | 0.097045 | 0.048020 | 1.572322e+00 | 1.414974 | 1.282404 | 0.097023 | 0.201385 | 0.056512 | 2.571047e-01 | 0.902977 | 0.916548 |
| comp8 | 0.102352 | 0.031881 | 1.406115e+00 | 2.065242 | 1.335352 | 0.100524 | 0.110604 | 0.158052 | 1.437481e-03 | 0.899476 | 0.949425 |
| comp9 | 0.106273 | 0.034356 | 1.495352e+00 | 1.664641 | 1.229181 | 0.103560 | 0.148219 | 0.175895 | 3.820012e-04 | 0.896440 | 0.954876 |

</div>


    Finished:
    03_Age_Sex
    /home/traaffneu/dansch/Documents/digital_rodent/digitalrodent/results/normative_model/Model_Testing/03_Age_Sex
