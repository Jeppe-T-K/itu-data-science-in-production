# Data Stages

---

## Medallion Architecture

The medallion architecture, popularized by Databricks, is a data organization pattern that structures data into quality layers.

<img src="/images/data-and-model-versioning/medallion-architecture.png" alt="Medallion Architecture" style="clip-path: inset(60px 0 00px 0);"/>

From https://www.databricks.com/glossary/medallion-architecture

<details><summary style="font-size: 1.2em;">🥉 Bronze Layer</summary>

- Raw, unprocessed data
- High volume, high velocity
- Schema may evolve over time
- No or minimal transformations

</details>

<details><summary style="font-size: 1.2em;">🥈 Silver Layer</summary>

- Data has been cleaned (nulls handled, outliers removed)
- Schema is more stable
- Some business logic applied
- Data scientists typical users

</details>

<details><summary style="font-size: 1.2em;">🥇 Gold Layer</summary>

- Enriched with business context
- Aggregated and summarized
- Optimized for querying
- Ready for analytics and ML

</details>


<details><summary>Version control where?</summary>

Version control can be applied at different stages:

- **Bronze**: Version the raw data files themselves
- **Silver**: Version the cleaning/transformation code
- **Gold**: Version the aggregation and business logic

Best practice: Version at all stages to ensure full reproducibility.
</details>

---

<details><summary style="font-size: 1.5em;">ML Datasets</summary>

<img src="/images/data-and-model-versioning/raw-cleansed-curated.webp" alt="Raw cleansed curated" style="clip-path: inset(0px 0 0px 0);"/>
From https://www.rawgeneration.com/products/best-juice-cleanse when searching on Google for "raw cleansed curated".

<details><summary style="font-size: 1.2em;">Raw data</summary>

- Unprocessed
- May contain errors
- May have inconsistent formats
- Often needs significant cleaning
- Bronze

</details>

<details><summary style="font-size: 1.2em;">Cleansed data</summary>

- Missing values handled
- Outliers identified or removed
- Duplicate records removed
- Consistent formats
- Silver

</details>

<details><summary style="font-size: 1.2em;">Curated/model-ready data</summary>

- Features engineered
- Categorical variables encoded
- Numerical features scaled
- Balanced classes (if applicable)
- Ready for model consumption
- Gold

</details>
</details>

---

<details><summary style="font-size: 1.5em;">Other Considerations</summary>

<details><summary style="font-size: 1.2em;">Train/validate/test split</summary>

![Train/Test Split](/images/data-and-model-versioning/kaggle-split.png)

From https://www.kaggle.com/discussions/general/516681

<details><summary>_Where in the Medallion structure would you do this?_</summary>
Between bronze and silver.
</details>
</details>

<details><summary style="font-size: 1.2em;">Controlling randomness</summary>

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
</details>
</details>

---

<details><summary style="font-size: 1.5em;">ML Pipelines</summary>

ML data pipelines move data through various processing stages to prepare it for model training.

![Data Pipeline](/images/data-and-model-versioning/data-pipeline.png)

From https://mlops-guide.github.io/Versionamento/pipelines_dvc/

1. **Preprocessing**: Extract, validate, split, feature engineering, normalization
2. **Training**: Train/run model, extract outputs 
3. **Postprocessing**: Format output, evaluate results, monitoring

→ More on this in future lectures

</details>

---

<details><summary style="font-size: 1.5em;">Best Practices</summary>

1. **Clear boundaries**: Each stage should have well-defined inputs and outputs
2. **Quality gates**: Validate data quality at each stage
3. **Documentation**: Document the purpose and transformations of each stage
4. **Testing**: Test transformations between stages
5. **DAGs**: Directed acyclic graphs -- watch out for infinite loops 
</details>