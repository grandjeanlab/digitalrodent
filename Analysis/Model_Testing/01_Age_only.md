# Model Testing - Age only


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

model_name = "01_Age_only"

response_vars = [f"comp{i}" for i in range(1, 19)]

covariates = age_cols


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
    ['Age(week)', 'Age(week)_Missing']

    Batch effects:
    []

    Rows:
    2020

    Missing covariate values:
    0

    Missing response values:
    0

    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.1311310918797663e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4004387 - 2026-10-10 11:58:10 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4004387 - 2026-10-10 11:58:10 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.7236451062454015e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.9719224250034764e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.17281972383748e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4004387 - 2026-10-10 11:58:11 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.7767117013021116e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.8158967127251304e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.831792563341103e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/scipy/optimize/_numdiff.py:687: RuntimeWarning: overflow encountered in divide
      df_dx = [delf / delx for delf, delx in zip(df, dx)]
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.6609364438061243e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.678320367909752e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4004387 - 2026-10-10 11:58:12 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.034713939804274e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.3125522794182174e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/scipy/optimize/_numdiff.py:687: RuntimeWarning: overflow encountered in divide
      df_dx = [delf / delx for delf, delx in zip(df, dx)]
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.642651625327014e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.389732954253618e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4004387 - 2026-10-10 11:58:12 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.293119838886687e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.420568458879582e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/scipy/optimize/_numdiff.py:687: RuntimeWarning: overflow encountered in divide
      df_dx = [delf / delx for delf, delx in zip(df, dx)]


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
| comp1 | 0.075618 | 0.047649 | 1.767560 | 1.183605 | 1.282516 | 0.075607 | 0.256831 | 0.109138 | 1.097436e-05 | 0.924393 | 0.922276 |
| comp10 | 0.062481 | 0.043193 | 1.171208 | 2.101277 | 1.324757 | 0.062355 | 0.107781 | 0.060798 | 1.450821e-02 | 0.937645 | 0.927704 |
| comp11 | 0.058919 | 0.036881 | 1.214676 | 1.756905 | 1.350814 | 0.058871 | 0.156396 | 0.039963 | 1.082999e-01 | 0.941129 | 0.934360 |
| comp12 | 0.063158 | 0.033416 | 1.134289 | 1.825062 | 1.356988 | 0.063156 | 0.146661 | 0.067609 | 6.550760e-03 | 0.936844 | 0.936905 |
| comp13 | 0.094520 | 0.037203 | 1.105526 | 2.104391 | 1.300883 | 0.094474 | 0.103099 | 0.126055 | 3.700924e-07 | 0.905526 | 0.943501 |
| comp14 | 0.048862 | 0.032797 | 1.219221 | 2.308653 | 1.363992 | 0.048808 | 0.091756 | 0.059793 | 1.622017e-02 | 0.951192 | 0.951861 |
| comp15 | 0.070283 | 0.034653 | 1.237970 | 1.815027 | 1.320057 | 0.070175 | 0.142233 | 0.071891 | 3.834030e-03 | 0.929825 | 0.933937 |
| comp16 | 0.065376 | 0.047030 | 1.738055 | 1.574641 | 1.306656 | 0.065367 | 0.178936 | 0.080190 | 1.253943e-03 | 0.934633 | 0.919079 |
| comp17 | 0.080099 | 0.036634 | 2.168804 | 1.924125 | 1.312069 | 0.080096 | 0.125841 | 0.063363 | 1.084225e-02 | 0.919904 | 0.939160 |
| comp18 | 0.039671 | 0.042327 | 1.653482 | 1.949055 | 1.361826 | 0.039545 | 0.131817 | 0.018541 | 4.563866e-01 | 0.960455 | 0.915542 |
| comp2 | 0.066461 | 0.036510 | 1.042727 | 1.752648 | 1.314446 | 0.066268 | 0.150857 | 0.065287 | 8.657556e-03 | 0.933732 | 0.936984 |
| comp3 | 0.069179 | 0.027921 | 0.992563 | 2.265865 | 1.379730 | 0.069177 | 0.096239 | 0.046507 | 6.160762e-02 | 0.930823 | 0.971624 |
| comp4 | 0.047627 | 0.051980 | 1.495182 | 1.578670 | 1.334799 | 0.047534 | 0.185044 | 0.011746 | 6.370581e-01 | 0.952466 | 0.900309 |
| comp5 | 0.075121 | 0.031807 | 1.536427 | 2.041488 | 1.310202 | 0.075022 | 0.112004 | 0.066508 | 7.484610e-03 | 0.924978 | 0.951767 |
| comp6 | 0.068187 | 0.032426 | 1.045266 | 2.155967 | 1.338036 | 0.068082 | 0.103093 | 0.075035 | 2.541906e-03 | 0.931918 | 0.954787 |
| comp7 | 0.066332 | 0.053342 | 1.569338 | 1.419437 | 1.295216 | 0.066297 | 0.206500 | 0.069835 | 4.975641e-03 | 0.933703 | 0.912875 |
| comp8 | 0.051096 | 0.034109 | 1.167248 | 2.101555 | 1.385434 | 0.051095 | 0.115177 | 0.252924 | 5.269900e-25 | 0.948905 | 0.937539 |
| comp9 | 0.057333 | 0.033540 | 1.256967 | 1.714116 | 1.338826 | 0.057047 | 0.161444 | 0.046930 | 5.927429e-02 | 0.942953 | 0.938853 |

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
| comp1 | 0.038769 | 0.046238 | 4.159134e+12 | 1.213855 | 1.274641 | 0.034290 | 0.252688 | -0.019147 | 7.012004e-01 | 0.965710 | 0.917421 |
| comp10 | 0.031304 | 0.047030 | 1.299860e+00 | 2.116977 | 1.339195 | 0.028932 | 0.109547 | -0.044569 | 3.715926e-01 | 0.971068 | 0.920441 |
| comp11 | 0.045049 | 0.038119 | 1.368444e+00 | 1.750936 | 1.357674 | 0.042089 | 0.159821 | -0.037565 | 4.514691e-01 | 0.957911 | 0.929065 |
| comp12 | 0.043146 | 0.044752 | 1.318029e+00 | 1.841959 | 1.324883 | 0.034583 | 0.141761 | -0.028659 | 5.657225e-01 | 0.965417 | 0.935231 |
| comp13 | 0.046561 | 0.042277 | 1.171043e+00 | 2.124175 | 1.292412 | 0.042629 | 0.103056 | -0.001530 | 9.755387e-01 | 0.957371 | 0.929650 |
| comp14 | 0.010533 | 0.036832 | 1.371500e+00 | 2.325901 | 1.343379 | 0.007503 | 0.090245 | -0.055439 | 2.662649e-01 | 0.992497 | 0.946899 |
| comp15 | 0.025754 | 0.041782 | 2.625966e+12 | 1.859619 | 1.312547 | 0.016452 | 0.138858 | -0.056376 | 2.582503e-01 | 0.983548 | 0.908932 |
| comp16 | 0.017825 | 0.049010 | 2.036012e+00 | 1.613087 | 1.294736 | 0.012492 | 0.174894 | -0.049614 | 3.198594e-01 | 0.987508 | 0.915603 |
| comp17 | 0.051317 | 0.051683 | 1.557380e+00 | 1.959877 | 1.309058 | 0.043968 | 0.123411 | -0.034083 | 4.945207e-01 | 0.956032 | 0.927635 |
| comp18 | 0.012550 | 0.054653 | 1.935818e+00 | 1.964782 | 1.321067 | 0.003332 | 0.126904 | -0.067925 | 1.730019e-01 | 0.996668 | 0.913097 |
| comp2 | 0.035005 | 0.040297 | 1.251346e+00 | 1.773588 | 1.308823 | 0.030355 | 0.149701 | -0.041029 | 4.108123e-01 | 0.969645 | 0.937404 |
| comp3 | 0.025907 | 0.031386 | 1.036059e+00 | 2.286606 | 1.404141 | 0.022411 | 0.098990 | -0.077889 | 1.180356e-01 | 0.977589 | 0.962881 |
| comp4 | 0.035324 | 0.062376 | 1.737001e+00 | 1.572627 | 1.345685 | 0.032994 | 0.189635 | -0.061745 | 2.155661e-01 | 0.967006 | 0.896803 |
| comp5 | 0.041791 | 0.041782 | 1.329516e+00 | 2.070169 | 1.332457 | 0.035124 | 0.113662 | -0.023382 | 6.393749e-01 | 0.964876 | 0.932730 |
| comp6 | 0.051246 | 0.042772 | 1.265070e+00 | 2.173295 | 1.368320 | 0.046351 | 0.105648 | 0.019084 | 7.021487e-01 | 0.953649 | 0.940069 |
| comp7 | 0.027391 | 0.055446 | 1.694308e+00 | 1.462315 | 1.329745 | 0.025846 | 0.209171 | -0.049321 | 3.227227e-01 | 0.974154 | 0.902709 |
| comp8 | 0.048085 | 0.040792 | 1.461017e+00 | 2.106463 | 1.376574 | 0.043836 | 0.114036 | 0.276615 | 1.573611e-08 | 0.956164 | 0.930798 |
| comp9 | 0.030179 | 0.039307 | 1.654623e+00 | 1.726945 | 1.291484 | 0.021948 | 0.154819 | -0.059479 | 2.329215e-01 | 0.978052 | 0.939119 |

</div>


    Finished:
    01_Age_only
    /home/traaffneu/dansch/Documents/digital_rodent/digitalrodent/results/normative_model/Model_Testing/01_Age_only
