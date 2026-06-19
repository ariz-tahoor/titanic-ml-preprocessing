# Titanic Dataset — Cleaning, Feature Engineering, Encoding & Scaling Pipeline

A complete preprocessing pipeline that takes the raw Kaggle Titanic dataset (891 rows × 12 columns) from "unusable for ML" to model-ready (891 rows × 16 columns), using pandas and scikit-learn.

## The Problem

Raw data straight from Kaggle isn't ML-ready:
- `Age`: 177 missing values (19.9% of rows)
- `Cabin`: 687 missing values (77.1% of rows — effectively unusable)
- `Embarked`: 2 missing values
- Categorical columns (`Sex`, `Embarked`) and free-text (`Name`) that models can't read as numbers
- `Fare` ranges from 0 to 512 — heavily skewed, with real outliers (first-class fares vs. steerage)
- No signal captured from family relationships or social title, despite both being predictive of survival

## What I Did

**1. Missing Value Handling**
- `Cabin`: dropped entirely — 77% missing made imputation unreliable
- `Age`: imputed with the median (robust to the column's skew, avoids inventing implausible ages)
- `Embarked`: imputed with the mode (only 2 missing, negligible impact either way)

**2. Feature Engineering**
- `FamilySize` = `SibSp` + `Parch` + 1 — collapses two raw counts into one feature representing total family aboard
- `Title` extracted from `Name` (Mr/Mrs/Miss/Master/etc.), then consolidated rare titles (Capt, Col, Dr, Rev, Major, Lady, Sir, etc.) into a single `Other` bucket to avoid sparse one-off categories

**3. Encoding**
- One-hot encoded `Sex`, `Title`, and `Embarked` using `pd.get_dummies(..., drop_first=True)`
- `drop_first` removes the redundant baseline category per feature, avoiding multicollinearity
- Result: 3 categorical columns → 6 binary columns (`Sex_male`, `Title_Mr`, `Title_Mrs`, `Title_Other`, `Embarked_Q`, `Embarked_S`)

**4. Scaling**
- Applied `StandardScaler` to `Age` and `FamilySize` — both reasonably close to normal, no major outlier distortion
- Applied `RobustScaler` (median/IQR-based) to `Fare` specifically, because its outliers (max $512 vs. median ~$14) would have skewed a standard mean/std scale and compressed the resolution between ordinary fares
- Concretely: with `StandardScaler`, the bulk of ordinary fares (25th–75th percentile, $7.91–$31) would have landed in a narrow band between z = -0.49 and z = -0.02 — barely distinguishable from each other. With `RobustScaler`, that same range spans -0.28 to 0.72, preserving real differences between typical passengers while still letting the $512 outlier register as clearly large (scaled value ≈ 21.6)
- This is the kind of per-column scaler decision that matters in practice — not just running every numeric column through the same transformer by default

## Before → After

| Column | Before | After |
|---|---|---|
| `Age` | 0.42–80, 19.9% missing | 0 missing, scaled (mean 0, std 1) |
| `Cabin` | 77.1% missing | dropped |
| `Sex` | "male"/"female" | `Sex_male` (0/1) |
| `Name` | free text | `Title_Mr`, `Title_Mrs`, `Title_Other` |
| `SibSp` + `Parch` | two raw counts | single `FamilySize` feature, scaled |
| `Embarked` | "S"/"C"/"Q", 2 missing | `Embarked_Q`, `Embarked_S` (0 missing) |
| `Fare` | 0–512.33, raw scale | RobustScaler: median 0, typical range -0.28 to 0.72, outlier preserved at ~21.6 |

**Dataset shape:** 891 rows × 12 columns → 891 rows × 16 columns

## Tools

`pandas` · `scikit-learn` (`StandardScaler`, `RobustScaler`) · `numpy`

## Why This Matters

This is the process I run for client datasets: audit missing data and choose a strategy per column (not blanket dropna), engineer features that capture signal the raw columns miss, encode categoricals to avoid both information loss and redundant dummy traps, and pick a scaler per column based on its actual distribution rather than applying one transformer to every numeric field. Same process, any dataset.

---
*Full code: [Titanic_preprocessing.ipynb](./Titanic_preprocessing.ipynb)*
