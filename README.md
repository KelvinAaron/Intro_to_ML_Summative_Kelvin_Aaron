# Predicting 30-Day Hospital Readmission for Diabetic Patients

Intro to Machine Learning summative project. A three-class classification task that compares a traditional ML model (XGBoost, with Logistic Regression and Random Forest baselines) against a deep learning model (MLP) on hospital encounter records.

## Problem

Given encounter-level information (demographics, diagnoses, lab tests, medications, prior visits), predict whether a diabetic patient will be readmitted:

| Class | Meaning | Share of data |
|---|---|---|
| `NO` | not readmitted | ~53% |
| `>30` | readmitted after more than 30 days | ~36% |
| `<30` | readmitted within 30 days | ~11% |

## Dataset

[Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (UCI Machine Learning Repository). 101,766 encounters, 50 columns, 71,518 unique patients. Original study: Strack et al. (2014), *BioMed Research International*.

The notebook downloads a copy of the UCI zip from Google Drive with `gdown` (the file ID is set in the first cells). The same zip is available from the UCI link above.

## Repository contents

| File | Description |
|---|---|
| `Intro_ML_Summative_KelvinAaron.ipynb` | Full pipeline: cleaning, preprocessing, models, tuning, saving |
| `requirements.txt` | Python dependencies |

The notebook also writes the two best models to Google Drive (`best_xgb_d3_lr0.05.joblib` and `best_mlp_64-32.h5`).

## Method

1. **Cleaning.** Dropped `weight` (97% missing), `encounter_id`, and `payer_code`. `"None"` in `A1Cresult` and `max_glu_serum` is kept as the category "Not measured", and `"?"` becomes "Unknown". Removed patients who died or went to hospice (they cannot be readmitted) and 3 invalid-gender rows. Dropped constant columns. Result: 99,340 rows.
2. **Features.** ICD-9 diagnosis codes grouped into 9 disease categories; age bracket converted to its midpoint; admission and discharge IDs treated as categorical.
3. **Split.** 70/15/15 train/validation/test, split **by patient** (`GroupShuffleSplit`, seed 42), so no patient appears in more than one split.
4. **Preprocessing.** Standard scaling for numeric columns, log transform plus scaling for the three visit-count columns, one-hot encoding for categoricals (categories seen fewer than 50 times pooled). Fitted on train only.
5. **Imbalance.** Class weights (`balanced`) for every model.
6. **Models.** Logistic Regression, 2 Random Forests, 2 XGBoost models, 2 MLPs (Keras), plus a 20-trial random search over XGBoost hyperparameters scored on the validation set.
7. **Tracking.** Each run is logged to a local results table and, optionally, to Weights & Biases.

## Results (validation set)

| Run | Macro-F1 | ROC-AUC (OvR) | Recall `<30` |
|---|---|---|---|
| Logistic Regression | 0.434 | 0.656 | 0.419 |
| Random Forest, depth 15 | 0.453 | 0.664 | 0.328 |
| Random Forest, depth 10 | 0.425 | 0.660 | 0.443 |
| XGBoost baseline | 0.453 | 0.665 | 0.376 |
| **XGBoost, depth 3 (saved)** | 0.444 | 0.665 | 0.414 |
| MLP 128-64 | 0.442 | 0.658 | 0.414 |
| **MLP 64-32 (saved)** | 0.442 | 0.663 | 0.385 |

Across all seven runs and the 20 search trials, validation AUC stays between 0.656 and 0.669. Linear, bagged, boosted, and neural models all reach about the same ceiling, which is consistent with the limit being in the data (no post-discharge information, strong class imbalance) and not in the model choice.

**Saved models.** XGBoost (depth 3, learning rate 0.05) and the 64-32 MLP were chosen because they matched the best validation AUC in their families with the smallest train/validation gaps (0.013 and 0.008 in macro-F1).

## Results (test set)

The two saved models were scored once on the held-out test patients, after all selection decisions were made.

| Model | Macro-F1 | Accuracy | ROC-AUC (OvR) | Recall `NO` | Recall `>30` | Recall `<30` |
|---|---|---|---|---|---|---|
| XGBoost, depth 3 | 0.446 | 0.507 | 0.670 | 0.610 | 0.375 | 0.434 |
| MLP 64-32 | 0.444 | 0.507 | 0.671 | 0.623 | 0.363 | 0.416 |

Test scores are in line with validation (AUC 0.670-0.671 against 0.663-0.665), so the roughly 0.66-0.67 ceiling holds on patients that were never used for training or model selection.

## How to run

1. Open the notebook in Google Colab (the notebook uses `google.colab` for Drive and secrets).
2. Mount Google Drive and create the folders `MyDrive/Intro_ML_Summative/results` and `MyDrive/Intro_ML_Summative/models` before running the search and saving cells.
3. Optional: to enable Weights & Biases, add a Colab secret named `WANDB_API_KEY`. Without it the notebook still runs and records results locally.
4. Run the cells from top to bottom. A GPU is not required.

To run elsewhere, install the dependencies with `pip install -r requirements.txt`, remove the Colab-specific cells (Drive mount and secrets), and point the file paths at local folders.

## Notes and limitations

- The test split (14,859 rows) was used once, for the two saved models only, after all tuning decisions. No model was changed afterwards.
- The fitted preprocessor was not saved. The saved models expect inputs produced by the notebook's `ColumnTransformer`, which can be rebuilt by rerunning the preprocessing cells with the same seed.
- Validation scores of a model chosen from many candidates are slightly optimistic. Differences of about 0.01 between runs are within the expected noise.
- Data covers US hospitals from 1999 to 2008 and may not reflect current practice.
