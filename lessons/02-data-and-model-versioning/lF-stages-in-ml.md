# Data Stages in ML

## Data Pipelines

ML data pipelines move data through various processing stages to prepare it for model training.

![Data Pipeline](/images/data-and-model-versioning/data-pipeline.png)

From https://mlops-guide.github.io/Versionamento/pipelines_dvc/

### Typical Pipeline Stages

1. **Ingestion**: Extract data from source systems
2. **Validation**: Check data quality and consistency
3. **Cleaning**: Handle missing values, outliers, duplicates
4. **Transformation**: Feature engineering, normalization, encoding
5. **Splitting**: Divide data into training, validation, test sets
6. **Storage**: Save processed data for model training

## Typical ML Datasets

### Raw Data

Data as it arrives from source systems.

**Characteristics:**
- Unprocessed
- May contain errors
- May have inconsistent formats
- Often needs significant cleaning

**Versioning:** Critical to version raw data to ensure reproducibility

### Cleansed Data

Data after cleaning and basic transformations.

**Characteristics:**
- Missing values handled
- Outliers identified or removed
- Duplicate records removed
- Consistent formats

**Versioning:** Version the cleaning code and resulting data

### Model-Ready Data

Data fully prepared for model training.

**Characteristics:**
- Features engineered
- Categorical variables encoded
- Numerical features scaled
- Balanced classes (if applicable)
- Ready for model consumption

**Versioning:** Version the feature engineering and final dataset

## Train/Test Split

Splitting data into training, validation, and test sets is a critical step in ML.

### Common Split Ratios

- 70/15/15 (train/validation/test)
- 80/10/10
- 60/20/20

The optimal ratio depends on:
- Dataset size (larger datasets can use smaller test sets)
- Model complexity
- Risk tolerance

### stratified Splitting

For classification problems, ensure class distribution is preserved in each split.

### Time-Based Splitting

For time series data, split by time rather than randomly:
- Train: Historical data up to time T
- Validation: Data from T to T+N
- Test: Data from T+N onwards

## Control Randomness with Seed

To ensure reproducibility, always set random seeds:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Set random seeds
np.random.seed(42)
pd.options.mode.chained_assignment = None  # default='warn'

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

```python
import tensorflow as tf

# For TensorFlow
tf.random.set_seed(42)
```

```python
import torch

# For PyTorch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

![Train/Test Split](/images/data-and-model-versioning/kaggle-split.png)

From https://www.kaggle.com/discussions/general/516681

## Versioning Train/Test Splits

It's important to version your splits to ensure:

1. **Reproducibility**: The same split can be recreated
2. **Consistency**: Different team members use the same splits
3. **Traceability**: Know which data version and split produced which model

### Approaches

1. **Store split indices**: Save the indices used for each split
2. **Store split files**: Save the actual split files
3. **Deterministic splits**: Use hashing or deterministic algorithms

## Best Practices

1. **Document splits**: Record the split strategy and ratios used
2. **Version everything**: Version raw data, cleaning code, and splits
3. **Validate splits**: Check that splits are representative of the overall data
4. **Use consistent seeds**: Set random seeds for all random operations
5. **Stratify**: For classification, maintain class balance in splits
