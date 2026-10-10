# Model Testing - Sex only


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

model_name = "02_Sex_only"

response_vars = [f"comp{i}" for i in range(1, 19)]

covariates = sex_cols


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
    ['Sex_M', 'Sex_Missing']

    Batch effects:
    []

    Rows:
    2020

    Missing covariate values:
    0

    Missing response values:
    0

    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.0862817803094005e-16.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4005598 - 2026-10-10 11:59:04 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.978896094523358e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.021090336337021e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.238038191216342e-16.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.6742252009861625e-16.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4005598 - 2026-10-10 11:59:05 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.136395558640526e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.355558762717389e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.61830285347706e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.801448654830269e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4005598 - 2026-10-10 11:59:05 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.7652857377457035e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.6473839450829512e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/scipy/optimize/_numdiff.py:687: RuntimeWarning: overflow encountered in divide
      df_dx = [delf / delx for delf, delx in zip(df, dx)]
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.5015717783313844e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4005598 - 2026-10-10 11:59:05 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.739509995702328e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.82350937928476e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.291033549998841e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.88454775351784e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4005598 - 2026-10-10 11:59:05 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.6184596279880483e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.679652418264248e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.5484643080546e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4005598 - 2026-10-10 11:59:06 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.783744754898193e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 4005598 - 2026-10-10 11:59:06 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.429707398023951e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/plotter.py:209: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all Axes decorations.
      plt.tight_layout()


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
| comp1 | 0.077962 | 0.056436 | 1.749197 | 1.188864 | 1.287774 | 0.077962 | 0.256504 | 0.202644 | 1.946190e-16 | 0.922038 | 0.900781 |
| comp10 | 0.056686 | 0.045668 | 1.241723 | 2.097373 | 1.320853 | 0.056686 | 0.108107 | 0.169199 | 7.601407e-12 | 0.943314 | 0.934794 |
| comp11 | 0.050595 | 0.039728 | 1.270230 | 1.750742 | 1.344652 | 0.050595 | 0.157082 | 0.137791 | 2.676483e-08 | 0.949405 | 0.938539 |
| comp12 | 0.047092 | 0.037129 | 1.171475 | 1.830635 | 1.362560 | 0.047092 | 0.147913 | 0.141151 | 1.209612e-08 | 0.952908 | 0.934103 |
| comp13 | 0.055112 | 0.045619 | 1.215490 | 2.161876 | 1.358368 | 0.055112 | 0.105316 | 0.118133 | 1.917980e-06 | 0.944888 | 0.931903 |
| comp14 | 0.060803 | 0.043069 | 1.250772 | 2.297097 | 1.352436 | 0.060803 | 0.091176 | 0.186219 | 4.480689e-14 | 0.939197 | 0.940284 |
| comp15 | 0.056821 | 0.047525 | 1.285452 | 1.828493 | 1.333523 | 0.056821 | 0.143251 | 0.159169 | 1.237890e-10 | 0.943179 | 0.928994 |
| comp16 | 0.063744 | 0.050371 | 1.712668 | 1.591335 | 1.323350 | 0.063744 | 0.179092 | 0.163925 | 3.368092e-11 | 0.936256 | 0.908272 |
| comp17 | 0.047944 | 0.050124 | 3.552462 | 1.968948 | 1.356892 | 0.047944 | 0.128021 | 0.136005 | 4.050996e-08 | 0.952056 | 0.928774 |
| comp18 | 0.059783 | 0.040842 | 1.655098 | 1.900345 | 1.313116 | 0.059783 | 0.130421 | 0.162015 | 5.707171e-11 | 0.940217 | 0.927936 |
| comp2 | 0.067834 | 0.041213 | 1.040870 | 1.755363 | 1.317161 | 0.067834 | 0.150730 | 0.198447 | 8.179212e-16 | 0.932166 | 0.940867 |
| comp3 | 0.076131 | 0.032624 | 1.110053 | 2.254500 | 1.368365 | 0.076131 | 0.095879 | 0.209083 | 2.021108e-17 | 0.923869 | 0.964988 |
| comp4 | 0.068374 | 0.051114 | 1.453215 | 1.523500 | 1.279628 | 0.068374 | 0.183009 | 0.195970 | 1.880335e-15 | 0.931626 | 0.915338 |
| comp5 | 0.060905 | 0.041213 | 1.989842 | 2.076462 | 1.345175 | 0.060905 | 0.112856 | 0.167132 | 1.369974e-11 | 0.939095 | 0.940928 |
| comp6 | 0.062289 | 0.039480 | 1.063643 | 2.177125 | 1.359194 | 0.062289 | 0.103413 | 0.135911 | 4.139773e-08 | 0.937711 | 0.948168 |
| comp7 | 0.056314 | 0.055446 | 1.563187 | 1.445664 | 1.321443 | 0.056314 | 0.207601 | 0.130461 | 1.417894e-07 | 0.943686 | 0.901856 |
| comp8 | 0.057346 | 0.037327 | 1.192837 | 2.079452 | 1.363330 | 0.057346 | 0.114797 | 0.171555 | 3.849119e-12 | 0.942654 | 0.934964 |
| comp9 | 0.058362 | 0.035644 | 1.280268 | 1.705916 | 1.330626 | 0.058362 | 0.161331 | 0.167733 | 1.155221e-11 | 0.941638 | 0.935687 |

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
| comp1 | 0.093352 | 0.053168 | 1.138908e+12 | 1.170152 | 1.230938 | 0.091497 | 0.245090 | 0.250171 | 3.505573e-07 | 0.908503 | 0.909739 |
| comp10 | 0.063851 | 0.048515 | 1.303130e+00 | 2.081175 | 1.303393 | 0.063801 | 0.107562 | 0.172428 | 4.992964e-04 | 0.936199 | 0.926934 |
| comp11 | 0.054509 | 0.047030 | 1.321378e+00 | 1.751876 | 1.358614 | 0.054189 | 0.158808 | 0.144607 | 3.581190e-03 | 0.945811 | 0.922118 |
| comp12 | 0.049088 | 0.043267 | 1.291802e+00 | 1.864557 | 1.347481 | 0.045473 | 0.140959 | 0.152476 | 2.117624e-03 | 0.954527 | 0.916284 |
| comp13 | 0.077216 | 0.044752 | 1.109058e+00 | 2.143184 | 1.311421 | 0.076639 | 0.101209 | 0.164045 | 9.346489e-04 | 0.923361 | 0.933135 |
| comp14 | 0.075698 | 0.030396 | 1.286418e+00 | 2.288649 | 1.306126 | 0.075405 | 0.087103 | 0.177684 | 3.320382e-04 | 0.924595 | 0.943565 |
| comp15 | 0.075741 | 0.056139 | 1.024097e+12 | 1.824332 | 1.277260 | 0.072967 | 0.134809 | 0.183408 | 2.101174e-04 | 0.927033 | 0.923562 |
| comp16 | 0.087380 | 0.044554 | 1.775126e+00 | 1.572749 | 1.254397 | 0.085678 | 0.168288 | 0.223210 | 5.903038e-06 | 0.914322 | 0.920314 |
| comp17 | 0.060017 | 0.052673 | 1.788017e+00 | 1.976663 | 1.325845 | 0.057401 | 0.122540 | 0.148302 | 2.806869e-03 | 0.942599 | 0.924693 |
| comp18 | 0.068611 | 0.039802 | 1.786227e+00 | 1.894966 | 1.251252 | 0.065723 | 0.122868 | 0.177882 | 3.268984e-04 | 0.934277 | 0.930207 |
| comp2 | 0.075260 | 0.035644 | 1.148466e+00 | 1.762161 | 1.297396 | 0.074831 | 0.146227 | 0.210965 | 1.907871e-05 | 0.925169 | 0.943168 |
| comp3 | 0.093043 | 0.033861 | 9.361294e-01 | 2.235586 | 1.353121 | 0.092518 | 0.095374 | 0.213017 | 1.574601e-05 | 0.907482 | 0.962760 |
| comp4 | 0.079405 | 0.053960 | 1.586327e+00 | 1.503038 | 1.276096 | 0.079327 | 0.185036 | 0.224249 | 5.326782e-06 | 0.920673 | 0.914380 |
| comp5 | 0.067923 | 0.051485 | 1.227557e+00 | 2.085759 | 1.348048 | 0.066361 | 0.111807 | 0.186816 | 1.589762e-04 | 0.933639 | 0.924635 |
| comp6 | 0.067561 | 0.040297 | 1.235011e+00 | 2.170164 | 1.365189 | 0.066691 | 0.104515 | 0.118479 | 1.719951e-02 | 0.933309 | 0.944890 |
| comp7 | 0.068314 | 0.050000 | 1.538203e+00 | 1.421388 | 1.288819 | 0.068288 | 0.204564 | 0.165349 | 8.493989e-04 | 0.931712 | 0.902520 |
| comp8 | 0.063338 | 0.036832 | 1.544660e+00 | 2.086621 | 1.356732 | 0.062280 | 0.112931 | 0.185355 | 1.792793e-04 | 0.937720 | 0.934062 |
| comp9 | 0.066658 | 0.043267 | 1.569908e+00 | 1.723829 | 1.288368 | 0.065041 | 0.151370 | 0.174033 | 4.413413e-04 | 0.934959 | 0.930842 |

</div>


    Finished:
    02_Sex_only
    /home/traaffneu/dansch/Documents/digital_rodent/digitalrodent/results/normative_model/Model_Testing/02_Sex_only
