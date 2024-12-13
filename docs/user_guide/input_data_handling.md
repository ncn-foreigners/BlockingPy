(input_data_handling)=
# Input Data Handling

##Supported Input Formats

BlockingPy is flexible in terms of input data formats. The package accepts three main types of input:

- Text Data: `pandas.Series` containing raw text
- Sparse Matrices: `scipy.sparse.csr_matrix` for pre-computed document-term matrices
- Dense Arrays: `numpy.ndarray` for numeric feature vectors

## Text Processing Options

When working with text data, you can configure preprocessing using the control_txt parameter:

```python
texts = pd.Series([
    "john smith",
    "smith john",
    "jane doe"
])

control_txt = {
    'n_shingles': 2,           # Size of character n-grams
    'max_features': 5000,      # Maximum number of features to keep
    'lowercase': True,         # Convert text to lowercase
    'strip_non_alphanum': True # Remove non-alphanumeric characters
}

result = blocker.block(x=texts, control_txt=control_txt)
```
## Dataframes

If you have a DataFrame with multiple columns (like name, address, etc.), we recommend combining these columns into a single text column before passing it to the blocker:

```python
import pandas as pd
from blockingpy import Blocker

# Example DataFrame with multiple columns
df = pd.DataFrame({
    'name': ['John Smith', 'Jane Doe', 'Smith John'],
    'city': ['New York', 'Boston', 'NYC'],
    'occupation': ['engineer', 'doctor', 'engineer']
})

# Combine relevant columns into a single text field
# You can adjust the separator and columns based on your needs (and also with control_txt to a degree)
df['blocking_key'] = df['name'] + ' ' + df['city'] + ' ' + df['occupation']

# Pass the combined text column to the blocker
blocker = Blocker()
result = blocker.block(x=df['blocking_key'])
```

## Pre-computed Document-Term Matrices

If you have already vectorized your text data or are working with numeric features, you can pass a sparse document-term matrix:

```python
from scipy import sparse

# Example sparse DTMs
dtm_1 = sparse.csr_matrix((n_docs, n_features))
dtm_2 = sparse.csr_matrix((n_docs_2, n_features_2))

# Column names are required for sparse matrices
feature_names_1 = [f'feature_{i}' for i in range(n_features)]
feature_names_2 = [f'feature_{i}' for i in range(n_features_2)]

result = blocker.block(
    x=dtm_1,
    y=dtm_2, 
    x_colnames=feature_names_1,
    y_colnames=feature_names_2,
)
```


## Dense Numeric Arrays
For dense feature vectors, use numpy arrays:
```python
import numpy as np

# Example feature matrix
features = np.array([
    [1.0, 2.0, 0.0],
    [2.0, 0.0, 0.0],
    [2.0, 1.0, 1.0]
])

# Column names are required for numpy arrays
feature_names = ['feat_1', 'feat_2', 'feat_3']

result = blocker.block(
    x=features, 
    x_colnames=feature_names
)
```

## Input Validation
BlockingPy performs several validations on input data:

- Format Checking: Ensures inputs are in supported formats
- Compatibility: Verifies feature compatibility between datasets
- Column Names: Validates presence of required column names
- Dimensions: Checks for appropriate matrix dimensions

If validation fails, clear error messages are provided indicating the issue.