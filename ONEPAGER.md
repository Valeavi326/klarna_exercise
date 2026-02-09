# Klarna Pay Later – 21-Day PD Model (1-Pager)

## Objective
The objective of this case study is to build an early-risk model that estimates the
probability that a Pay Later loan will default within 21 days from issuance. The focus
is not on maximizing predictive performance, but on demonstrating sound reasoning,
robust validation, and production-ready modeling choices.

---

## Target definition
The target is defined as:

**default_21d = 1 if amount_outstanding_21d > 0, else 0**

Post-issuance information is used exclusively to label the outcome and is never included
among model features. All model inputs are strictly available at, or before, loan issuance.

---

## Leakage control and validation strategy
To prevent information leakage and overly optimistic performance estimates:
- All post-issuance variables (e.g. outstanding amounts at 14d or 21d) are explicitly
  excluded from the feature set.
- Model evaluation relies on a strict **time-based split**, ensuring that training data
  always precedes validation and test data in time.
- TimeSeriesSplit cross-validation is used on the training window to assess stability
  across different temporal segments.

This setup reflects how the model would behave in a real production environment.

---

## Feature engineering rationale

### Delta features (behavioural dynamics)
Rather than relying only on absolute cumulative values, several variables are expressed
as **deltas between time windows** (e.g. repayment or failed-payment activity between
6m and 3m, or exposure growth between 14d and 7d).

This choice is motivated by two considerations:
- Behavioural **trajectories** (improving vs deteriorating behaviour) are often more
  informative than static levels in early credit risk.
- Delta features reduce redundancy and collinearity between overlapping time windows,
  leading to more stable models and clearer interpretation.

For each signal, a short-term level is retained (e.g. last 7d or 14d), while longer
windows are represented through deltas.

---

### Data-consistency flags and monotonicity checks
Some variables are expected, by construction, to follow monotonic relationships
(e.g. cumulative exposure or payments over longer windows should not be smaller than
shorter ones).

When such conditions are violated:
- Observations are **not dropped**, to avoid introducing selection bias.
- Conservative fixes are applied where appropriate.
- Explicit **binary flags** are created and retained as model features.

These flags allow the model to learn whether data inconsistencies themselves carry
risk information, while preserving transparency and auditability.

---

### First-time borrower handling
The variable `days_since_first_loan = -1` is treated as a valid signal indicating a
first-time borrower, rather than as an error. A dedicated indicator is introduced to
separate lifecycle effects from tenure length, improving interpretability and stability.

---

## Model choice and feature selection
A LightGBM classifier is trained within a scikit-learn pipeline with standard imputation
and encoding steps.

Specific feature selection choices include:
- **Inclusion of delta-based behavioural features**, as described above.
- **Exclusion of highly granular calendar variables** (e.g. day of issuance), which were
  found to capture short-term operational or promotional effects rather than intrinsic
  customer risk, potentially harming out-of-time stability.
- Retention of coarser temporal signals only when they add robustness rather than noise.

Predicted probabilities are calibrated using sigmoid (Platt) calibration to ensure that
PD estimates are suitable for threshold-based decisioning.

---

## Performance and decisioning
Performance is evaluated on a time-based holdout set and shows:
- Stable discrimination (ROC-AUC ≈ 0.65).
- Precision-Recall performance above the base default rate.
- Strong calibration (low Brier score), supporting the use of probabilities rather than
hard classifications.

Rather than optimizing a single metric, model outputs are translated into business-level
artifacts (deciles and threshold tables) that quantify approval rates, observed default
rates, and defaulter capture.

### Validation metrics and class imbalance

The default rate in the dataset is low (around 4–5%), resulting in a strongly
imbalanced classification problem. In this setting, standard metrics such as
accuracy are not informative, and ROC-AUC alone can provide an incomplete picture
of model performance.

While ROC-AUC measures the model’s ability to rank positive observations above
negative ones across all thresholds, it is largely insensitive to class imbalance
and does not reflect performance in the low-false-positive region that is most
relevant for credit decisioning.

For this reason, Precision-Recall AUC (PR-AUC) is also evaluated. PR-AUC focuses
explicitly on the positive (default) class and measures how well the model
concentrates true defaulters among high-risk predictions. In highly imbalanced
datasets, PR-AUC provides a more meaningful assessment of the model’s practical
usefulness than ROC-AUC alone.

Both metrics are therefore reported: ROC-AUC to assess overall ranking quality,
and PR-AUC to evaluate performance on the rare but business-critical default class.


### Probability calibration

In a credit risk context, predicted probabilities are often used directly for
decisioning, threshold selection, and expected loss estimation. For this reason,
good ranking performance alone is not sufficient: predicted probabilities must
also be well calibrated.

Tree-based models such as LightGBM are known to produce poorly calibrated raw
probability estimates, even when ranking performance is strong. To address this,
sigmoid (Platt) calibration is applied on the training data.

Calibration ensures that predicted PDs can be interpreted as meaningful probabilities:
for example, that loans predicted with a 10% PD actually default at approximately
that rate when observed in aggregate. This improves the reliability of threshold-based
approval strategies and makes model outputs suitable for downstream business use,
rather than only for relative ranking.

---

## Deliverables
The final solution includes:
- A fully reproducible training and evaluation pipeline.
- Analysis artifacts supporting model understanding.
- A FastAPI service that can be hosted locally and queried via HTTP.









