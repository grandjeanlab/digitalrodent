# Digital Rodent Partial BLR Model


``` python
from pathlib import Path
import pandas as pd

from pcntoolkit import (
    BLR,
    NormativeModel,
    NormData,
    plot_qq
)

import pcntoolkit.util.output


print("PYTHON CHUNK STARTED")


# ============================================================
# HARD-CODED PATHS
# ============================================================

input_file = Path(
    "/home/traaffneu/dansch/Documents/digital_rodent/digitalrodent/"
    "results/combined_dataset/Partial_Model/Partial_Model_dataframe.csv"
)

model_dir = Path(
    "/home/traaffneu/dansch/Documents/digital_rodent/digitalrodent/"
    "results/normative_model/Model_Testing/Partial_Model"
)

model_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading:", input_file)

model_df = pd.read_csv(input_file)

print("Loaded rows:", len(model_df))
print("Columns:")
print(model_df.columns.tolist())

pcntoolkit.util.output.Output.set_show_messages(False)


# ============================================================
# MODEL VARIABLES
# ============================================================

response_vars = [
    "comp2",
    "comp10",
    "comp11"
]

batch_effects = []

covariates = [
    col for col in model_df.columns
    if col not in (
        [
            "Scan_id",
            "Participant_Id",
            "Session",
            "Dataset_name"
        ]
        + response_vars
    )
]

print("\nResponses:")
print(response_vars)

print("\nCovariates:")
print(covariates)

print("\nBatch effects:")
print(batch_effects)


# ============================================================
# CHECK DATA
# ============================================================

print(
    "\nMissing covariates:",
    model_df[covariates].isna().sum().sum()
)

print(
    "Missing responses:",
    model_df[response_vars].isna().sum().sum()
)

print("\nUnique values per covariate:")
for col in covariates:
    print(col, model_df[col].nunique(dropna=False))


# ============================================================
# NORMDATA
# ============================================================

print("\nCreating NormData...")

norm_data = NormData.from_dataframe(
    name="DigitalRodent_Partial",
    dataframe=model_df,
    covariates=covariates,
    batch_effects=batch_effects,
    response_vars=response_vars,
    remove_Nan=False
)

print("NormData created")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("Creating train/test split...")

train, test = norm_data.train_test_split()

print("Train/test split created")


# ============================================================
# MODEL
# ============================================================

print("Creating BLR model...")

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

print("BLR model created")


# ============================================================
# FIT
# ============================================================

print("\nSTARTING FIT")

model.fit_predict(
    train,
    test
)

print("FIT FINISHED")


# ============================================================
# QQ
# ============================================================

qq_dir = model_dir / "plots" / "qq"
qq_dir.mkdir(parents=True, exist_ok=True)

plot_qq(
    test,
    plot_id_line=True,
    save_dir=str(qq_dir)
)


# ============================================================
# STATISTICS
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

print("\nDONE")
print("Saved to:")
print(model_dir)
```

    PYTHON CHUNK STARTED
    Loading: /home/traaffneu/dansch/Documents/digital_rodent/digitalrodent/results/combined_dataset/Partial_Model/Partial_Model_dataframe.csv
    Loaded rows: 1640
    Columns:
    ['Scan_id', 'Participant_Id', 'Session', 'Dataset_name', 'Age(week)', 'MRI', 'MRI T.R.', 'Sex_encoded', 'Coil_encoded', 'Ventilation_encoded', 'Rodent.strain_F1 C6/129P', 'Rodent.strain_ICR', 'fMRI.sequence_ME-EPI', 'fMRI.sequence_SE-EPI', 'Anesthesia_Dex', 'Anesthesia_Dexiso', 'Anesthesia_Hal', 'Anesthesia_Iso', 'Anesthesia_Med', 'Anesthesia_Mediso', 'Anesthesia_Propofol', 'comp2', 'comp10', 'comp11']

    Responses:
    ['comp2', 'comp10', 'comp11']

    Covariates:
    ['Age(week)', 'MRI', 'MRI T.R.', 'Sex_encoded', 'Coil_encoded', 'Ventilation_encoded', 'Rodent.strain_F1 C6/129P', 'Rodent.strain_ICR', 'fMRI.sequence_ME-EPI', 'fMRI.sequence_SE-EPI', 'Anesthesia_Dex', 'Anesthesia_Dexiso', 'Anesthesia_Hal', 'Anesthesia_Iso', 'Anesthesia_Med', 'Anesthesia_Mediso', 'Anesthesia_Propofol']

    Batch effects:
    []

    Missing covariates: 0
    Missing responses: 0

    Unique values per covariate:
    Age(week) 36
    MRI 6
    MRI T.R. 9
    Sex_encoded 2
    Coil_encoded 2
    Ventilation_encoded 2
    Rodent.strain_F1 C6/129P 2
    Rodent.strain_ICR 2
    fMRI.sequence_ME-EPI 2
    fMRI.sequence_SE-EPI 2
    Anesthesia_Dex 2
    Anesthesia_Dexiso 2
    Anesthesia_Hal 2
    Anesthesia_Iso 2
    Anesthesia_Med 2
    Anesthesia_Mediso 2
    Anesthesia_Propofol 2

    Creating NormData...
    NormData created
    Creating train/test split...
    Train/test split created
    Creating BLR model...
    BLR model created

    STARTING FIT

    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 1152305 - 2026-10-10 15:21:41 - Estimation of posterior distribution failed due to: 
    A singular matrix detected: slice(s) [0] are singular.
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.0899946206201367e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/util/output.py:239: UserWarning: Process: 1152305 - 2026-10-10 15:21:41 - Estimation of posterior distribution failed due to: 
    Matrix is not positive definite
      warnings.warn(message)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.951072219377268e-23.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.6254959422189753e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.8154440650252013e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.344326263190269e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.124735488092826e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.869969606721251e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.519039936482912e-23.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.499503637698696e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.6578189460048127e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/dansch/.conda/envs/digitalrodent/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:674: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.632336908423673e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)

    FIT FINISHED

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
| comp10 | 0.685245 | 0.024085 | 0.326605 | 1.276935 | 0.531300 | 0.684846 | 0.064447 | 0.878337 | 0.0 | 0.315154 | 0.971570 |
| comp11 | 0.679449 | 0.037530 | 0.390582 | 1.043912 | 0.661952 | 0.678362 | 0.093662 | 0.862881 | 0.0 | 0.321638 | 0.943528 |
| comp2 | 0.674652 | 0.032317 | 0.309631 | 1.029075 | 0.612764 | 0.674011 | 0.091109 | 0.862497 | 0.0 | 0.325989 | 0.954689 |

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
| comp10 | 0.724636 | 0.045122 | 0.371612 | 1.265216 | 0.478690 | 0.723019 | 0.057997 | 0.874575 | 1.668601e-104 | 0.276981 | 0.972501 |
| comp11 | 0.722447 | 0.054268 | 0.433193 | 1.024437 | 0.621800 | 0.715998 | 0.086211 | 0.867073 | 1.134477e-100 | 0.284002 | 0.954716 |
| comp2 | 0.711563 | 0.037683 | 0.346122 | 1.105424 | 0.610116 | 0.709670 | 0.079451 | 0.861745 | 4.335637e-98 | 0.290330 | 0.923041 |

</div>


    DONE
    Saved to:
    /home/traaffneu/dansch/Documents/digital_rodent/digitalrodent/results/normative_model/Model_Testing/Partial_Model
