# Wild-type and 3xTG-AD both unseen: deviation scores (log DR, warped BLR)


The 3xTG study has two groups: wild-type (WT) controls and 3xTG-AD mice.
Normally the WT controls (`Dataset_name == "3xTG"`) are part of the
training set. Here the normative model is **refit without them**, so
that **both** groups are unseen, exactly like a new lab sending its
data:

- **WT controls** test generalization: healthy mice from a lab the model
  has never seen should get z ~ N(0, 1) (mean ~ 0, SD ~ 1, ~5 % beyond
  ±1.96).
- **3xTG-AD vs WT** tests the clinical question: do the deviation scores
  separate the two groups better than the raw DR values?

The model settings are the same as `Model_Training_log.qmd` (B version):
log(DR), warped BLR (SinhArcsinh), heteroskedastic, B-splines on age, FD
and log tSNR, no batch effects.

- With **PCNToolkit ≥ 1.3** the log is done by the toolkit
  (`y_transform="log"`).
- With 1.1.2 it is done by hand; the z-scores are identical.

Results go to `results/normative_model_WT_heldout_log/`. The older
`Model_Alzheimer_Comparison.qmd` (no warp, no log) and its folder are
left untouched.

``` python
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, roc_curve

from pcntoolkit import BLR, BsplineBasisFunction, NormativeModel, NormData
import pcntoolkit.util.output

from component_labels import COMPONENTS, ACRONYM, NAME, SELECTED_COMPONENTS

warnings.filterwarnings("ignore", category=FutureWarning)
# mixed models with ~2 scans per animal often hit the variance boundary
warnings.filterwarnings("ignore", category=ConvergenceWarning)
pcntoolkit.util.output.Output.set_show_messages(False)
plt.style.use("default")   # pcntoolkit turns on seaborn's dark grid at import

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
# Sex as a batch effect (fixed_effect + fixed_effect_var), as suggested by
# Konstantinos. False = B version (no batch effects at all).
# Only changes anything with PCNToolkit >= 1.3. Keep False until Jo agrees.
SEX_AS_BATCH_EFFECT = False

PCN_VERSION = version("pcntoolkit")
USE_Y_TRANSFORM = tuple(int(v) for v in PCN_VERSION.split(".")[:2]) >= (1, 3)
print(f"PCNToolkit {PCN_VERSION}; log done by",
      "y_transform='log'" if USE_Y_TRANSFORM else "hand (np.log)")
if SEX_AS_BATCH_EFFECT and not USE_Y_TRANSFORM:
    print("NOTE: sex batch effect is only run with PCNToolkit >= 1.3")

# networks in the figures: the main selection + dorsal hippocampus (comp11)
FIG_COMPONENTS = SELECTED_COMPONENTS[:-1] + ["comp11", SELECTED_COMPONENTS[-1]]

root = Path.cwd().parent
tag = "_sexBE" if (SEX_AS_BATCH_EFFECT and USE_Y_TRANSFORM) else ""
out_dir = root / "results" / f"normative_model_WT_heldout_log{tag}"
fig_dir = out_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
WT_COLOR = "#2a78d6"
AD_COLOR = "#eb6834"
RAW_COLOR = "#6b6a64"
Z_COLOR = "#6250d6"
INK = "#2b2b29"
MUTED = "#6b6a64"

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 6, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "svg.fonttype": "none",
})
```

    PCNToolkit 1.4.0; log done by y_transform='log'

## Refit the normative model without the wild-type controls

``` python
model_df = pd.read_csv(root / "results" / "combined_dataset" / "Model_dataframe.csv")

id_cols = ["Scan_id", "Participant_Id", "Session", "Dataset_name"]
covariates = [c for c in model_df.columns if c not in id_cols + COMPONENTS]
assert (model_df[COMPONENTS].dropna() > 0).all().all(), "log needs positive DR values"

batch_effects = []
if SEX_AS_BATCH_EFFECT and USE_Y_TRANSFORM:
    model_df["Sex"] = np.where(model_df["Sex_encoded"] == 1, "M", "F")
    covariates.remove("Sex_encoded")   # sex moves from covariate to batch effect
    batch_effects = ["Sex"]


def prepare_y(df):
    """log(DR) by hand only for PCNToolkit < 1.3 (otherwise y_transform does it)."""
    df = df.copy()
    if not USE_Y_TRANSFORM:
        df[COMPONENTS] = np.log(df[COMPONENTS])
    return df


def to_normdata(name, df):
    return NormData.from_dataframe(
        name=name, dataframe=prepare_y(df), covariates=covariates,
        batch_effects=batch_effects, response_vars=COMPONENTS, remove_Nan=False,
    )


train_df = (
    model_df[model_df["Dataset_name"] != "3xTG"]
    .dropna(subset=covariates + COMPONENTS)
)

spline_columns = [covariates.index(c) for c in ["Age(week)", "FD average", "log_tSNR"]]

blr_kwargs = dict(
    basis_function_mean=BsplineBasisFunction(basis_column=spline_columns, degree=3, nknots=5),
    basis_function_var=BsplineBasisFunction(basis_column=spline_columns, degree=3, nknots=5),
    heteroskedastic=True,
    warp_name="WarpSinhArcsinh",
)
if batch_effects:
    blr_kwargs.update(fixed_effect=True, fixed_effect_var=True)

model_kwargs = dict(
    inscaler="standardize", outscaler="standardize",
    savemodel=True, evaluate_model=False, saveresults=False, saveplots=False,
    save_dir=str(out_dir),
)
if USE_Y_TRANSFORM:
    model_kwargs["y_transform"] = "log"

model = NormativeModel(BLR(**blr_kwargs), **model_kwargs)
model.fit(to_normdata("train_without_WT", train_df))
print(f"Model refit on {len(train_df)} scans from "
      f"{train_df['Dataset_name'].nunique()} datasets; 3xTG WT controls held out")
```

    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.1592251886301825e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/util/output.py:342: UserWarning: Process: 1456915 - 2026-11-28 17:43:01 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.213991381482309e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.799762886339258e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.3667185922202613e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.184499691093532e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.096731518432099e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.143447513557264e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.28081072856408e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.7283941210708457e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.866793256207851e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.469967787355982e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.1909698853044265e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.8002282933726706e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.1816169654654575e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.9777683978408546e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.757737311257318e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.9440449831841195e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.1776084493484065e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.9696205232976535e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.489087615346546e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:615: RuntimeWarning: overflow encountered in exp
      beta = np.exp(Phi_var.dot(w_d))
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:696: RuntimeWarning: invalid value encountered in dot
      self.A = XtLambda_n.dot(X) + self.Lambda_a
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 0.0.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:759: RuntimeWarning: divide by zero encountered in log
      logdetSigma_n = -np.sum(np.log(self.lambda_n_vec))
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/numpy/_core/fromnumeric.py:83: RuntimeWarning: invalid value encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:615: RuntimeWarning: overflow encountered in exp
      beta = np.exp(Phi_var.dot(w_d))
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:696: RuntimeWarning: invalid value encountered in dot
      self.A = XtLambda_n.dot(X) + self.Lambda_a
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 0.0.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:615: RuntimeWarning: overflow encountered in exp
      beta = np.exp(Phi_var.dot(w_d))
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/math_functions/warp.py:579: RuntimeWarning: overflow encountered in sinh
      y = np.sinh(b * np.arcsinh(x) - a)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:696: RuntimeWarning: invalid value encountered in dot
      self.A = XtLambda_n.dot(X) + self.Lambda_a
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 0.0.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:759: RuntimeWarning: divide by zero encountered in log
      logdetSigma_n = -np.sum(np.log(self.lambda_n_vec))
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/numpy/_core/fromnumeric.py:83: RuntimeWarning: invalid value encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/math_functions/warp.py:621: RuntimeWarning: overflow encountered in cosh
      dx = (b * np.cosh(b * np.arcsinh(x) - a)) / np.sqrt(1 + x**2)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.5358906395718967e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/util/output.py:342: UserWarning: Process: 1456915 - 2026-11-28 17:43:08 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.5079501720914973e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.252722240048159e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.968140057306137e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.5182674852763946e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.079467666441359e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.461969501106213e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.565802017963935e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.690542889668749e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.2624342883990834e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.303521956115776e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.852664476448837e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.4729892527334484e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.7945770974544206e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.5702977723628e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.485592512738953e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.6612412273212615e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.5916798485285567e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.0627256416598008e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.2817723795161557e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:615: RuntimeWarning: overflow encountered in exp
      beta = np.exp(Phi_var.dot(w_d))
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:696: RuntimeWarning: invalid value encountered in dot
      self.A = XtLambda_n.dot(X) + self.Lambda_a
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 0.0.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:759: RuntimeWarning: divide by zero encountered in log
      logdetSigma_n = -np.sum(np.log(self.lambda_n_vec))
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/numpy/_core/fromnumeric.py:83: RuntimeWarning: invalid value encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.755128842784659e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/util/output.py:342: UserWarning: Process: 1456915 - 2026-11-28 17:43:12 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.7911473349805716e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.311578324395351e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.1982101746786189e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.843632059953665e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.454552949965442e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.123668710289511e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.739013794413302e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.521454189948603e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.3035463291188943e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.5239782486982744e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.911566750841824e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.027861258948049e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.454010127389132e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.1138126775417214e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.873096775464392e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.673555165129225e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/util/output.py:342: UserWarning: Process: 1456915 - 2026-11-28 17:43:12 - Posterior estimation failed: 
    A singular matrix detected: slice(s) [0] are singular.. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.0450163684251244e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.998730605530647e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.76790169823533e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/util/output.py:342: UserWarning: Process: 1456915 - 2026-11-28 17:43:14 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.25501553363312e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.832563807889963e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.256433888246864e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.0845665152989078e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.884211094395296e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.000278203896586e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.0952460509833406e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.078930841852073e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.044919988640002e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.241229270956628e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.1365832545372963e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.645888893552212e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.560774562983568e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.283677660221794e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.669705489032482e-23.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.112081343565262e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.348237149570211e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.1377329973540935e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.3261371500220304e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.2279075493091194e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/util/output.py:342: UserWarning: Process: 1456915 - 2026-11-28 17:43:15 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.8059800809704876e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.1861003326375279e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.38615014869481e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.4171984711287174e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.437841866874981e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.012524248187004e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.7679127176881794e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.269883528644308e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.088337565017054e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.242979898132861e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.7798938673266365e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.900737578002857e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.781983647980416e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.05450029566069e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.0167612528191434e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.606867594324694e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.384416791320318e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.8905471663966335e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.5231695613934153e-23.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:615: RuntimeWarning: overflow encountered in exp
      beta = np.exp(Phi_var.dot(w_d))
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 0.0.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:700: RuntimeWarning: invalid value encountered in multiply
      self.m = (invAXt * self.lambda_n_vec).dot(y)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.1117619428845084e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/util/output.py:342: UserWarning: Process: 1456915 - 2026-11-28 17:43:18 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.695042355488113e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.686891924026158e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.694113298645395e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.4358420841370484e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.269041588263988e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.505878148007389e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.082701848413896e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.3904268201456137e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.5369070508859513e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.189405227688927e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.485764046297386e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.1228006401448867e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.98084380354022e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.728469283075282e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.2582042078252453e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.133420776620251e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.58073783317989e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.6999298812403245e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.2658143256887537e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:615: RuntimeWarning: overflow encountered in exp
      beta = np.exp(Phi_var.dot(w_d))
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:696: RuntimeWarning: invalid value encountered in dot
      self.A = XtLambda_n.dot(X) + self.Lambda_a
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 0.0.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:615: RuntimeWarning: overflow encountered in exp
      beta = np.exp(Phi_var.dot(w_d))
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:696: RuntimeWarning: invalid value encountered in dot
      self.A = XtLambda_n.dot(X) + self.Lambda_a
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 0.0.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:759: RuntimeWarning: divide by zero encountered in log
      logdetSigma_n = -np.sum(np.log(self.lambda_n_vec))
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/numpy/_core/fromnumeric.py:83: RuntimeWarning: invalid value encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:615: RuntimeWarning: overflow encountered in exp
      beta = np.exp(Phi_var.dot(w_d))
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:696: RuntimeWarning: invalid value encountered in dot
      self.A = XtLambda_n.dot(X) + self.Lambda_a
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 0.0.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:759: RuntimeWarning: divide by zero encountered in log
      logdetSigma_n = -np.sum(np.log(self.lambda_n_vec))
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/numpy/_core/fromnumeric.py:83: RuntimeWarning: invalid value encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.8592789233074405e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/util/output.py:342: UserWarning: Process: 1456915 - 2026-11-28 17:43:24 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.667207056048511e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.6655390765100934e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.408660701947275e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.0854691471286873e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.039304202856169e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.496423099883089e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.646537125635655e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.3108808976901784e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.9477528228733574e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.916142224737578e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.629779617686529e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.750721518761188e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.796551816897518e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.474749983234047e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.36136391380766e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.154841738577828e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.088610016369503e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.3415646172478657e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.516936239635487e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 7.450883130874172e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/util/output.py:342: UserWarning: Process: 1456915 - 2026-11-28 17:43:28 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.939641056758014e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.1608839605841364e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.194726778585709e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.88510016180306e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.457165944860708e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.527975535376126e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.3496036914303536e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.771277111851415e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.0210060631642672e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.315612145847068e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.598735596888147e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.1146446252064643e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.143106472195885e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.0744522959196775e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.439317451205404e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.6719063644437069e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.313365369470355e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.405267809742678e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.5715448540861104e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.998911684871034e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/util/output.py:342: UserWarning: Process: 1456915 - 2026-11-28 17:43:29 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.231157348969247e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.2595990270610716e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.6259622268169613e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.338655626436449e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.9057755643341344e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.591956586197906e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 9.47908405092373e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.7907598060450757e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.49441365490984e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.3596662639231083e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.46431222019523e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.604131318555123e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.053436824822061e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.946716331052815e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.0630451902958843e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.567952650469636e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.5982656430295455e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.4347007154162136e-21.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/pcntoolkit/regression_model/blr.py:697: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.7893933542281303e-22.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/xarray/computation/apply_ufunc.py:820: RuntimeWarning: overflow encountered in exp
      result_data = func(*input_data)
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/xarray/computation/apply_ufunc.py:820: RuntimeWarning: overflow encountered in exp
      result_data = func(*input_data)

    Model refit on 1886 scans from 13 datasets; 3xTG WT controls held out

    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/xarray/computation/apply_ufunc.py:820: RuntimeWarning: overflow encountered in exp
      result_data = func(*input_data)

## Covariates of both groups

Both groups come from the same study (11.75 T, cryo coil, GE-EPI,
medetomidine + isoflurane, high dose), so the **protocol** covariates
are taken from the WT rows of `Model_dataframe.csv` for both groups.
Only age, FD, tSNR and sex are per scan.

`Alzheimer.csv` says *Ventilation = No* while the 3xTG metadata of the
WT controls says *y*. This is the same protocol, so the WT value is used
for both. Otherwise the two groups would differ in a covariate and that
difference would leak into the z-scores. Check this with the lab.

``` python
wt = model_df[model_df["Dataset_name"] == "3xTG"].copy()
wt["Group"] = "Wild-type"

alz_raw = pd.read_csv(root / "data" / "Alzheimer" / "Alzheimer.csv")
assert (alz_raw["Anesthesia"].str.lower() == "mediso").all()
assert (alz_raw["Anesthesia level"].str.lower() == "high").all()

per_scan = ["Age(week)", "FD average", "log_tSNR", "Sex_encoded"]
protocol_cols = [c for c in covariates if c not in per_scan]
protocol = wt[protocol_cols].iloc[0]
assert (wt[protocol_cols] == protocol).all().all(), "WT rows differ in protocol"

ad = pd.DataFrame(index=alz_raw.index)
ad["Age(week)"] = pd.to_numeric(alz_raw["Age(week)"])
ad["FD average"] = pd.to_numeric(alz_raw["FD average"])
ad["log_tSNR"] = np.log(pd.to_numeric(alz_raw["tSNR"]))
sex = alz_raw["Sex"].astype(str).str.strip().str.upper().str[0]
ad["Sex_encoded"] = sex.map({"F": 0.0, "M": 1.0})
for c in protocol_cols:
    ad[c] = protocol[c]

vent_alz = alz_raw["Ventilation"].astype(str).str.lower().str[0].eq("y").astype(float)
if "Ventilation_encoded" in protocol_cols and (vent_alz != protocol["Ventilation_encoded"]).any():
    print(f"WARNING: Alzheimer.csv ventilation = {sorted(alz_raw['Ventilation'].unique())}, "
          f"WT metadata = {protocol['Ventilation_encoded']:.0f}; using the WT value for both")

for c in COMPONENTS + ["Scan_id", "Participant_Id", "Session"]:
    ad[c] = alz_raw[c]
ad["Dataset_name"] = "3xTG-AD"
ad["Group"] = "3xTG-AD"
if batch_effects:
    ad["Sex"] = np.where(ad["Sex_encoded"] == 1, "M", "F")
assert ad[covariates].notna().all().all(), "missing covariates in the 3xTG-AD scans"

both = pd.concat([wt, ad], ignore_index=True)
both["Animal"] = both["Group"] + "_" + both["Participant_Id"].astype(str)
both["Age"] = np.where(both["Age(week)"] < 20, "3 mo", "6 mo")
both["AD"] = (both["Group"] == "3xTG-AD").astype(int)

print(both.groupby(["Group", "Age"]).agg(scans=("Scan_id", "size"),
                                         animals=("Animal", "nunique")))
print("\nPer-scan covariates by group (median):")
print(both.groupby("Group")[["Age(week)", "FD average", "log_tSNR"]].median().round(3))
```

    WARNING: Alzheimer.csv ventilation = ['No'], WT metadata = 1; using the WT value for both
                    scans  animals
    Group     Age                 
    3xTG-AD   3 mo     13       13
              6 mo      9        9
    Wild-type 3 mo     11       11
              6 mo     10       10

    Per-scan covariates by group (median):
               Age(week)  FD average  log_tSNR
    Group                                     
    3xTG-AD         13.0       0.014     3.790
    Wild-type       13.0       0.013     3.936

## Deviation scores

``` python
norm_both = to_normdata("WT_vs_3xTG", both[covariates + batch_effects + COMPONENTS])
model.compute_zscores(norm_both)

z = pd.DataFrame(norm_both.Z.values,
                 columns=[str(r) for r in norm_both.Z.coords["response_vars"].values])
for c in COMPONENTS:
    both[f"{c}_Z"] = z[c].to_numpy()

both.to_csv(out_dir / "WT_vs_3xTG_raw_and_z.csv", index=False)
```

## Generalization check: are the unseen WT controls N(0, 1)?

``` python
rows = []
for grp in ["Wild-type", "3xTG-AD"]:
    g = both[both.Group == grp]
    for c in COMPONENTS:
        zz = g[f"{c}_Z"]
        rows.append(dict(Group=grp, Component=c, Network=ACRONYM[c],
                         mean_z=zz.mean(), sd_z=zz.std(),
                         pct_beyond_196=100 * (zz.abs() > 1.96).mean()))
calib = pd.DataFrame(rows)
calib.round(3).to_csv(out_dir / "z_summary_by_group.csv", index=False)

summary = calib.groupby("Group")[["mean_z", "sd_z", "pct_beyond_196"]].agg(["mean", "min", "max"])
print("Over the 18 networks (ideal for WT: mean 0, SD 1, ~5 % beyond ±1.96):")
print(summary.round(2).to_string())
print("\nSelected networks:")
print(calib[calib.Component.isin(FIG_COMPONENTS)]
      .pivot(index="Network", columns="Group", values=["mean_z", "sd_z"]).round(2))
```

    Over the 18 networks (ideal for WT: mean 0, SD 1, ~5 % beyond ±1.96):
              mean_z              sd_z             pct_beyond_196           
                mean   min   max  mean   min   max           mean  min   max
    Group                                                                   
    3xTG-AD    -0.45 -1.46  0.29  0.40  0.08  0.66           0.51  0.0  9.09
    Wild-type  -0.35 -1.22  0.24  0.33  0.12  0.53           0.26  0.0  4.76

    Selected networks:
                  mean_z              sd_z          
    Group        3xTG-AD Wild-type 3xTG-AD Wild-type
    Network                                         
    Cg1 + RS       -0.82     -0.42    0.63      0.44
    Dorsal HP/SC   -1.11     -1.05    0.33      0.35
    M1             -0.20     -0.09    0.39      0.23
    S1uLp + S2      0.20      0.24    0.39      0.40
    V2 + RSD       -0.14     -0.48    0.50      0.26
    Ventral HP     -0.39     -0.36    0.08      0.12

QQ plot of the unseen WT controls (all 18 networks pooled, and the
3xTG-AD mice for comparison):

``` python
fig, ax = plt.subplots(figsize=(60 * MM, 60 * MM))
for grp, color in [("Wild-type", WT_COLOR), ("3xTG-AD", AD_COLOR)]:
    zz = np.sort(both.loc[both.Group == grp, [f"{c}_Z" for c in COMPONENTS]].to_numpy().ravel())
    theo = norm.ppf((np.arange(1, len(zz) + 1) - 0.5) / len(zz))
    ax.scatter(theo, zz, s=3, color=color, alpha=0.6, lw=0, label=f"{grp} (18 networks pooled)")
lim = [-4, 4]
ax.plot(lim, lim, "--", color=MUTED, lw=0.7)
ax.set(xlim=lim, ylim=lim, xlabel="Theoretical N(0,1) quantile", ylabel="Observed z")
ax.legend(frameon=False, loc="upper left")
ax.set_title("Both groups unseen by the model")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(fig_dir / "qq_WT_and_3xTG_unseen.svg", bbox_inches="tight")
fig.savefig(fig_dir / "qq_WT_and_3xTG_unseen.png", dpi=300, bbox_inches="tight")
plt.show()
```

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

![](Model_Alzheimer_WT_vs_3xTG_files/figure-commonmark/cell-7-output-2.png)

## Age centiles for the 3xTG protocol, with both groups

The curves are computed for **this study’s own protocol** (11.75 T,
cryo, med + iso high, study-median FD and tSNR), not for the Awake
reference mouse. Both groups can then be compared to the same curve.

``` python
age_grid = np.linspace(8, 30, 100)
grid = pd.DataFrame({c: [protocol[c]] * len(age_grid) for c in protocol_cols})
grid["Age(week)"] = age_grid
grid["FD average"] = both["FD average"].median()
grid["log_tSNR"] = both["log_tSNR"].median()
grid["Sex_encoded"] = 1.0                     # all mice in this study are male
if batch_effects:
    grid["Sex"] = "M"
for c in COMPONENTS:
    grid[c] = 1.0                             # placeholder Y (log(1) = 0), not used
grid_nd = to_normdata("grid_3xTG_protocol", grid[covariates + batch_effects + COMPONENTS])
CENTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
model.compute_centiles(grid_nd, centiles=CENTILES)
cent = grid_nd["centiles"]
if not USE_Y_TRANSFORM:                       # hand log: back to DR units
    cent = np.exp(cent)

fig, axes = plt.subplots(1, len(FIG_COMPONENTS), figsize=(36 * MM * len(FIG_COMPONENTS), 45 * MM))
for ax, comp in zip(axes, FIG_COMPONENTS):
    c = cent.sel(response_vars=comp)
    ax.fill_between(age_grid, c.sel(centile=0.05), c.sel(centile=0.95), color="#e4e3dd", lw=0)
    ax.fill_between(age_grid, c.sel(centile=0.25), c.sel(centile=0.75), color="#cfcdc5", lw=0)
    ax.plot(age_grid, c.sel(centile=0.5), color=INK, lw=1)
    for grp, color, dx in [("Wild-type", WT_COLOR, -0.6), ("3xTG-AD", AD_COLOR, 0.6)]:
        g = both[both.Group == grp]
        ax.scatter(g["Age(week)"] + dx, g[comp], s=7, color=color, lw=0, alpha=0.85,
                   label=f"{grp} (n={len(g)})")
    ax.set_title(ACRONYM[comp])
    ax.set_xlabel("Age (weeks)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("DR value")
axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
fig.suptitle("Normative centiles (5/25/50/75/95) for the 3xTG protocol; "
             "both groups unseen by the model", fontsize=7, color=INK)
fig.tight_layout()
fig.savefig(fig_dir / "centiles_3xTG_protocol_WT_and_AD.svg", bbox_inches="tight")
fig.savefig(fig_dir / "centiles_3xTG_protocol_WT_and_AD.png", dpi=300, bbox_inches="tight")
plt.show()
```

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

![](Model_Alzheimer_WT_vs_3xTG_files/figure-commonmark/cell-8-output-2.png)

## Figure 4A: group difference, raw values vs deviation scores

``` python
def density(ax, values, color, label):
    values = values.dropna().to_numpy(dtype=float)
    if len(values) < 3 or np.std(values) == 0:
        return
    pad = 0.25 * (values.max() - values.min())
    x = np.linspace(values.min() - pad, values.max() + pad, 300)
    y = gaussian_kde(values)(x)
    ax.plot(x, y, color=color, lw=1.6, label=label)
    ax.fill_between(x, 0, y, color=color, alpha=0.12, lw=0)
    ax.axvline(np.mean(values), color=color, lw=1, ls="--")


n_wt = both.loc[both.Group == "Wild-type", "Animal"].nunique()
n_ad = both.loc[both.Group == "3xTG-AD", "Animal"].nunique()

fig, axes = plt.subplots(2, len(FIG_COMPONENTS), figsize=(36 * MM * len(FIG_COMPONENTS), 70 * MM))
for j, comp in enumerate(FIG_COMPONENTS):
    for i, (suffix, xlabel) in enumerate([("", "DR value"), ("_Z", "Deviation score (z)")]):
        ax = axes[i, j]
        density(ax, both.loc[both.Group == "Wild-type", comp + suffix], WT_COLOR, f"Wild-type ({n_wt} mice)")
        density(ax, both.loc[both.Group == "3xTG-AD", comp + suffix], AD_COLOR, f"3xTG-AD ({n_ad} mice)")
        ax.set_xlabel(xlabel)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        if i == 0:
            ax.set_title(ACRONYM[comp], color=INK)
        if j == 0:
            ax.set_ylabel(["Raw values", "Deviation scores"][i], color=INK)
axes[0, -1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
fig.suptitle("Wild-type vs 3xTG-AD, both unseen: raw values (top) and deviation scores (bottom)\n"
             "dashed line = group mean; all scans of both ages", fontsize=7, color=INK)
fig.tight_layout()
fig.savefig(fig_dir / "figure4A_WT_vs_3xTG_distributions.svg", bbox_inches="tight")
fig.savefig(fig_dir / "figure4A_WT_vs_3xTG_distributions.png", dpi=300, bbox_inches="tight")
plt.show()
```

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

![](Model_Alzheimer_WT_vs_3xTG_files/figure-commonmark/cell-9-output-2.png)

## Figure 4B: group testing and classification

The group tests use a mixed model (value ~ group + age, random intercept
per animal). The SVM is a linear SVM on all 18 networks,
leave-one-animal-out, with a 200-permutation test at the animal level.

``` python
def group_test(col):
    d = both[[col, "AD", "Age", "Animal"]].rename(columns={col: "y"}).dropna()
    d["y"] = (d["y"] - d["y"].mean()) / d["y"].std()
    fit = smf.mixedlm("y ~ AD + C(Age)", d, groups=d["Animal"]).fit(reml=True)
    return fit.params["AD"], fit.pvalues["AD"]


def cohen_d(col):
    a, b = both.loc[both.AD == 1, col], both.loc[both.AD == 0, col]
    pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled


rows = []
for comp in COMPONENTS:
    beta_raw, p_raw = group_test(comp)
    beta_z, p_z = group_test(comp + "_Z")
    rows.append(dict(Component=comp, Network=ACRONYM[comp],
                     beta_raw=beta_raw, p_raw=p_raw, d_raw=cohen_d(comp),
                     beta_z=beta_z, p_z=p_z, d_z=cohen_d(comp + "_Z")))
group_diff = pd.DataFrame(rows)
group_diff["pFDR_raw"] = multipletests(group_diff["p_raw"], method="fdr_bh")[1]
group_diff["pFDR_z"] = multipletests(group_diff["p_z"], method="fdr_bh")[1]
group_diff.round(4).to_csv(out_dir / "group_differences_raw_vs_z.csv", index=False)
print(group_diff.round(3).to_string(index=False))


def svm_scores(columns, labels):
    X = both[columns].to_numpy()
    groups = both["Animal"].to_numpy()
    scores = np.zeros(len(labels))
    for tr, te in LeaveOneGroupOut().split(X, labels, groups):
        if len(np.unique(labels[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
        clf.fit(X[tr], labels[tr])
        scores[te] = clf.decision_function(X[te])
    return scores


rng = np.random.default_rng(0)
animals = both.drop_duplicates("Animal")[["Animal", "AD"]]
svm = {}
for name, columns in [("Raw", COMPONENTS), ("Deviation (z)", [c + "_Z" for c in COMPONENTS])]:
    y = both["AD"].to_numpy()
    scores = svm_scores(columns, y)
    auc = roc_auc_score(y, scores)
    null = []
    for _ in range(200):
        permuted = dict(zip(animals["Animal"], rng.permutation(animals["AD"].to_numpy())))
        y_perm = both["Animal"].map(permuted).to_numpy()
        null.append(roc_auc_score(y_perm, svm_scores(columns, y_perm)))
    p = (1 + np.sum(np.array(null) >= auc)) / (1 + len(null))
    svm[name] = dict(scores=scores, auc=auc, p=p)
    print(f"SVM {name}: AUC = {auc:.2f}, permutation p = {p:.3f}")
pd.DataFrame({k: dict(AUC=v["auc"], p_perm=v["p"]) for k, v in svm.items()}).T \
    .round(3).to_csv(out_dir / "svm_raw_vs_z.csv")

fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 62 * MM),
                         gridspec_kw=dict(width_ratios=[0.7, 1, 1.3]))
ax = axes[0]
n_sig = [(group_diff["pFDR_raw"] < 0.05).sum(), (group_diff["pFDR_z"] < 0.05).sum()]
ax.bar([0, 1], n_sig, color=[RAW_COLOR, Z_COLOR], width=0.55)
for x, v in zip([0, 1], n_sig):
    ax.text(x, v + 0.2, str(v), ha="center", va="bottom", color=INK)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Raw", "Deviation (z)"])
ax.set_ylim(0, max(n_sig + [1]) + 2)
ax.set_ylabel("Networks with group difference\n(FDR p < 0.05, of 18)")
ax.set_title("Group difference testing")

ax = axes[1]
for name, color in [("Raw", RAW_COLOR), ("Deviation (z)", Z_COLOR)]:
    fpr, tpr, _ = roc_curve(both["AD"], svm[name]["scores"])
    ax.plot(fpr, tpr, color=color, lw=1.6,
            label=f"{name}: AUC {svm[name]['auc']:.2f} (p {svm[name]['p']:.3f})")
ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=0.7)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend(frameon=False, loc="lower right")
ax.set_title("SVM classification, 18 networks\n(leave-one-animal-out)")

ax = axes[2]
order = group_diff.sort_values("d_z").reset_index(drop=True)
ypos = np.arange(len(order))
ax.hlines(ypos, order["d_raw"], order["d_z"], color="#d6d5cf", lw=1.2)
ax.scatter(order["d_raw"], ypos, color=RAW_COLOR, s=12, zorder=2, label="Raw")
ax.scatter(order["d_z"], ypos, color=Z_COLOR, s=12, zorder=3, label="Deviation (z)")
ax.axvline(0, color=MUTED, lw=0.7)
ax.set_yticks(ypos)
ax.set_yticklabels(order["Network"])
ax.set_xlabel("Cohen's d (3xTG-AD − wild-type)")
ax.legend(frameon=False, loc="lower right")
ax.set_title("Effect size per network")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(fig_dir / "figure4B_raw_vs_z_comparison.svg", bbox_inches="tight")
fig.savefig(fig_dir / "figure4B_raw_vs_z_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
```

    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/statsmodels/base/model.py:589: SingularMatrixWarning: Random effects covariance is singular
      return -self.score(params, *args) / nobs
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/statsmodels/base/model.py:589: SingularMatrixWarning: Random effects covariance is singular
      return -self.score(params, *args) / nobs
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/statsmodels/base/model.py:589: SingularMatrixWarning: Random effects covariance is singular
      return -self.score(params, *args) / nobs
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/statsmodels/base/model.py:589: SingularMatrixWarning: Random effects covariance is singular
      return -self.score(params, *args) / nobs
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/statsmodels/base/model.py:589: SingularMatrixWarning: Random effects covariance is singular
      return -self.score(params, *args) / nobs
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/statsmodels/base/model.py:589: SingularMatrixWarning: Random effects covariance is singular
      return -self.score(params, *args) / nobs
    /home/traaffneu/giuvas/.conda/envs/pcn14/lib/python3.12/site-packages/statsmodels/base/model.py:589: SingularMatrixWarning: Random effects covariance is singular
      return -self.score(params, *args) / nobs

    Component           Network  beta_raw  p_raw  d_raw  beta_z   p_z    d_z  pFDR_raw  pFDR_z
        comp1               Pir    -0.659  0.081 -0.945  -1.134 0.000 -1.670     0.146   0.000
        comp2                M1     0.284  0.383  0.274  -0.285 0.355 -0.359     0.492   0.533
        comp3        S1uLp + S2     0.346  0.290  0.333  -0.049 0.879 -0.108     0.402   0.901
        comp4        S1 HL + FL     0.631  0.073  0.551   0.229 0.538  0.127     0.145   0.648
        comp5     S1 FL + S1 BF     0.856  0.002  0.895   0.147 0.540  0.055     0.011   0.648
        comp6             S1 BF     1.197  0.000  1.275   0.820 0.023  0.702     0.001   0.103
        comp7          V2 + RSD     1.256  0.000  1.439   0.913 0.009  0.852     0.000   0.057
        comp8 AUd + AUv/TeA-ECT     0.027  0.939 -0.048  -0.268 0.427 -0.330     0.939   0.592
        comp9               Cg2     0.283  0.501  0.053  -0.389 0.300 -0.613     0.601   0.491
       comp10          Cg1 + RS     0.041  0.916 -0.104  -0.546 0.146 -0.727     0.939   0.353
       comp11      Dorsal HP/SC     0.830  0.003  0.879  -0.114 0.655 -0.190     0.012   0.737
       comp12        Ventral HP     0.634  0.023  0.638  -0.279 0.269 -0.359     0.070   0.485
       comp13               CPu     0.722  0.060  0.565  -0.371 0.177 -0.466     0.134   0.353
       comp14          CPu + En    -0.587  0.057 -0.589  -1.190 0.000 -1.474     0.134   0.000
       comp15       Septum + VP     0.092  0.812 -0.040  -0.604 0.076 -0.740     0.913   0.273
       comp16          Amygdala     0.430  0.222  0.382  -0.432 0.104 -0.535     0.333   0.311
       comp17          Thalamus     1.283  0.000  1.610  -0.056 0.901 -0.010     0.000   0.901
       comp18           MO + GI     0.413  0.213  0.413  -0.405 0.163 -0.457     0.333   0.353
    SVM Raw: AUC = 0.99, permutation p = 0.005
    SVM Deviation (z): AUC = 0.91, permutation p = 0.005

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

![](Model_Alzheimer_WT_vs_3xTG_files/figure-commonmark/cell-10-output-4.png)
