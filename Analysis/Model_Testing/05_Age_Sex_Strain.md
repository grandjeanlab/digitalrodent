# Model Testing - Age + Sex + Strain


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

model_name = "05_Age_Sex_Strain"

response_vars = [f"comp{i}" for i in range(1, 19)]

covariates = age_cols + sex_cols + strain_cols


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
    ['Age(week)', 'Age(week)_Missing', 'Sex_M', 'Sex_Missing', 'Rodent.strain_F1 C6/129P', 'Rodent.strain_ICR', 'Rodent.strain_Missing']

    Batch effects:
    []

    Rows:
    2020

    Missing covariate values:
    0

    Missing response values:
    0

    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.5566177281528457e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4008243 - 2026-10-10 12:01:04 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4008243 - 2026-10-10 12:01:04 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.0122640839950866e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.722241525721856e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.339290639723166e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.248885823187124e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.57067048924338e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.71683945260392e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.547697265812979e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.685038745024656e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.382723229552482e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.376931424879589e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.744857181076339e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.005328901633278e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4008243 - 2026-10-10 12:01:05 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.3614883052176626e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4008243 - 2026-10-10 12:01:05 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.473058907001403e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.142295614539785e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.39876706154828e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.3316020120464005e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.063094274992561e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.2597739611063659e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4008243 - 2026-10-10 12:01:08 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.5875069127485525e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4008243 - 2026-10-10 12:01:08 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.09545825595848e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.921719227736066e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.6303762951371087e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.078907817891603e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.3284900985188757e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.565678468827302e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.7599123431157437e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4008243 - 2026-10-10 12:01:09 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.059876444560433e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4008243 - 2026-10-10 12:01:09 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.0969889119747649e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4008243 - 2026-10-10 12:01:09 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.1559951464986717e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4008243 - 2026-10-10 12:01:09 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)


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
| comp1 | 0.496450 | 0.036262 | 1.005725 | 0.778712 | 0.877622 | 0.496141 | 0.189615 | 0.676821 | 5.601056e-217 | 0.503859 | 0.945745 |
| comp10 | 0.241694 | 0.048267 | 0.841239 | 1.815401 | 1.038881 | 0.240797 | 0.096985 | 0.584937 | 5.724835e-149 | 0.759203 | 0.919776 |
| comp11 | 0.187677 | 0.045297 | 0.873462 | 1.487594 | 1.081503 | 0.184213 | 0.145609 | 0.556580 | 4.111447e-132 | 0.815787 | 0.911521 |
| comp12 | 0.286376 | 0.044554 | 0.784566 | 1.572367 | 1.104293 | 0.284680 | 0.128154 | 0.598644 | 9.912058e-158 | 0.715320 | 0.903425 |
| comp13 | 0.407539 | 0.041337 | 0.713084 | 1.791812 | 0.988304 | 0.406857 | 0.083442 | 0.653177 | 3.417133e-197 | 0.593143 | 0.942628 |
| comp14 | 0.328748 | 0.039233 | 0.801405 | 2.003299 | 1.058638 | 0.328026 | 0.077122 | 0.620737 | 8.859317e-173 | 0.671974 | 0.938093 |
| comp15 | 0.319061 | 0.047401 | 0.830713 | 1.516157 | 1.021186 | 0.319055 | 0.121718 | 0.611826 | 1.447690e-166 | 0.680945 | 0.911687 |
| comp16 | 0.424202 | 0.037748 | 1.021149 | 1.222062 | 0.954077 | 0.423918 | 0.140482 | 0.667520 | 5.616674e-209 | 0.576082 | 0.917840 |
| comp17 | 0.305181 | 0.046411 | 1.455782 | 1.637418 | 1.025362 | 0.304923 | 0.109387 | 0.600370 | 7.291269e-159 | 0.695077 | 0.916954 |
| comp18 | 0.278471 | 0.049381 | 1.108252 | 1.651215 | 1.063986 | 0.278460 | 0.114252 | 0.597552 | 5.126500e-157 | 0.721540 | 0.884602 |
| comp2 | 0.281285 | 0.048144 | 0.725394 | 1.493085 | 1.054883 | 0.281283 | 0.132353 | 0.602220 | 4.370518e-160 | 0.718717 | 0.910531 |
| comp3 | 0.266257 | 0.028465 | 0.695216 | 2.008324 | 1.122189 | 0.263811 | 0.085588 | 0.567949 | 1.128638e-138 | 0.736189 | 0.966995 |
| comp4 | 0.238424 | 0.049257 | 1.027188 | 1.230326 | 0.986455 | 0.237742 | 0.165540 | 0.594212 | 7.510469e-155 | 0.762258 | 0.917444 |
| comp5 | 0.331547 | 0.041832 | 1.036427 | 1.775707 | 1.044421 | 0.331358 | 0.095228 | 0.612247 | 7.443487e-167 | 0.668642 | 0.935497 |
| comp6 | 0.356551 | 0.041708 | 0.687599 | 1.881736 | 1.063804 | 0.355641 | 0.085724 | 0.644677 | 1.653074e-190 | 0.644359 | 0.938709 |
| comp7 | 0.373795 | 0.044059 | 0.954785 | 1.095868 | 0.971648 | 0.373364 | 0.169170 | 0.661357 | 7.812913e-204 | 0.626636 | 0.909009 |
| comp8 | 0.061948 | 0.036139 | 0.879794 | 1.818011 | 1.101890 | 0.049509 | 0.115273 | 0.567659 | 1.671686e-138 | 0.950491 | 0.934244 |
| comp9 | 0.256522 | 0.046782 | 0.875546 | 1.428932 | 1.053642 | 0.255656 | 0.143438 | 0.575460 | 3.759961e-143 | 0.744344 | 0.922379 |

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
| comp1 | 0.523748 | 0.034356 | 1.144426e+12 | 0.769405 | 0.830191 | 0.521794 | 0.177815 | 0.660058 | 6.852251e-52 | 0.478206 | 0.944022 |
| comp10 | 0.215764 | 0.049010 | 8.984619e-01 | 1.803148 | 1.025366 | 0.214705 | 0.098513 | 0.553922 | 7.286404e-34 | 0.785295 | 0.916053 |
| comp11 | 0.183446 | 0.041584 | 9.664455e-01 | 1.463199 | 1.069937 | 0.177820 | 0.148066 | 0.534254 | 3.407427e-31 | 0.822180 | 0.915957 |
| comp12 | 0.313754 | 0.047723 | 8.761998e-01 | 1.568633 | 1.051557 | 0.303991 | 0.120367 | 0.597856 | 1.632581e-40 | 0.696009 | 0.933428 |
| comp13 | 0.418338 | 0.039109 | 7.126058e-01 | 1.806532 | 0.974769 | 0.416594 | 0.080449 | 0.645315 | 5.926428e-49 | 0.583406 | 0.948370 |
| comp14 | 0.342411 | 0.031683 | 8.590639e-01 | 2.031174 | 1.048652 | 0.341097 | 0.073531 | 0.614639 | 2.465136e-43 | 0.658903 | 0.928904 |
| comp15 | 0.343543 | 0.043267 | 1.047190e+12 | 1.571550 | 1.024478 | 0.341316 | 0.113635 | 0.603132 | 2.207657e-41 | 0.658684 | 0.846609 |
| comp16 | 0.421096 | 0.046040 | 1.127805e+00 | 1.298011 | 0.979660 | 0.419023 | 0.134148 | 0.647321 | 2.413900e-49 | 0.580977 | 0.900915 |
| comp17 | 0.291319 | 0.049010 | 1.032013e+00 | 1.662083 | 1.011264 | 0.287183 | 0.106563 | 0.577518 | 2.607511e-37 | 0.712817 | 0.929402 |
| comp18 | 0.292914 | 0.053465 | 1.219076e+00 | 1.711241 | 1.067526 | 0.291365 | 0.107007 | 0.594648 | 5.411396e-40 | 0.708635 | 0.844255 |
| comp2 | 0.288595 | 0.047525 | 8.155870e-01 | 1.582964 | 1.118199 | 0.288388 | 0.128245 | 0.582282 | 4.852538e-38 | 0.711612 | 0.877908 |
| comp3 | 0.249068 | 0.031881 | 7.036765e-01 | 2.051186 | 1.168721 | 0.243091 | 0.087103 | 0.550127 | 2.463789e-33 | 0.756909 | 0.960230 |
| comp4 | 0.243263 | 0.046535 | 1.107462e+00 | 1.216928 | 0.989986 | 0.242602 | 0.167828 | 0.599728 | 8.061415e-41 | 0.757398 | 0.907878 |
| comp5 | 0.340063 | 0.047525 | 8.330309e-01 | 1.797888 | 1.060176 | 0.337976 | 0.094149 | 0.609560 | 1.833665e-42 | 0.662024 | 0.909695 |
| comp6 | 0.350671 | 0.047525 | 7.964854e-01 | 1.871438 | 1.066462 | 0.348017 | 0.087354 | 0.642847 | 1.772728e-48 | 0.651983 | 0.929221 |
| comp7 | 0.376486 | 0.040099 | 9.775318e-01 | 1.147157 | 1.014587 | 0.376343 | 0.167364 | 0.635262 | 4.834233e-47 | 0.623657 | 0.894957 |
| comp8 | 0.099297 | 0.036832 | 1.026894e+00 | 1.816360 | 1.086470 | 0.076165 | 0.112092 | 0.586496 | 1.071314e-38 | 0.923835 | 0.951501 |
| comp9 | 0.272242 | 0.042574 | 1.041646e+00 | 1.506611 | 1.071150 | 0.267748 | 0.133960 | 0.558466 | 1.660470e-34 | 0.732252 | 0.882687 |

</div>


    Finished:
    05_Age_Sex_Strain
    /home/traaffneu/dansch/Documents/digital_rodent/digitalrodent/results/normative_model/Model_Testing/05_Age_Sex_Strain
