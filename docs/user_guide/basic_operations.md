(basic_operations)=
# Basic Operations

## Overview

BlockingPy provides three main operations:

- Record Linkage: Finding matching records between two datasets
- Deduplication: Finding duplicate records within a single dataset
- Evaluation: Evaluating blocking when true blocks are known (for both record linkage and deduplication) either inside the `block` method or separate `eval` method.

This guide covers the basic usage patterns for these operations.

## Record Linkage

### Basic usage

```python
from blockingpy import Blocker
import pandas as pd

# Example datasets
dataset1 = pd.Series([
    "john smith new york",
    "janee doe Boston",
    "robert brow chicagoo"
])

dataset2 = pd.Series([
    "smith john ny",
    "jane doe boston",
    "rob brown chicago"
])

# Initialize blocker
blocker = Blocker()

# Perform blocking
blocking_result = blocker.block(
    x=dataset1,  # Reference dataset
    y=dataset2,  # Query dataset
    ann="hnsw"   # Choose ANN algorithm (`hnsw` here)
)
```

## Results

The blocking operation returns a BlockingResult object with several useful attributes:

```python
# print blocking results
print(blocking_result)
# Shows:
# - Number of blocks created
# - Number of features created for blocking from text representation
# - Reduction ratio (how much the comparison space was reduced)
# - Distribution of block sizes

# Access detailed results
blocking_result.result  # DataFrame with columns: x, y, block, dist
blocking_result.method  # ANN algorithm used
blocking_result.colnames  # Features used for blocking
```

## Deduplication

### Basic Usage

```python
data = pd.Series([
    "john smith new york",
    "smith john ny",
    "jane doe boston",
    "j smith new york",
    "jane doe boston ma"
])

# Perform deduplication
result = blocker.block(
    x=data,
    ann="voyager"
)
```

Printing result gives similar results as in record linkage

## Evaluating Blocking Quality

If you have ground truth data, you can evaluate blocking quality: 

### Example ground truth for deduplication

```python
data = # your data

true_blocks = pd.DataFrame({
    'x': [0, 1, 2, 3, 4],      # Record indices
    'block': [0, 0, 1, 1, 1]   # True block assignments
})

result = blocker.block(
    x=data,
    true_blocks=true_blocks
)

# Access evaluation metrics
print(result.metrics)    # Shows precision, recall, F1-score, etc.
print(result.confusion)  # Confusion matrix
```
or alternatively with the use of `eval` method:

```python
data = # your data

true_blocks = pd.DataFrame({
    'x': [0, 1, 2, 3, 4],  
    'block': [0, 0, 1, 1, 1]   
})

result = blocker.block(
    x=data,
)
evals = blocker.eval(
    blocking_result=result,
    true_blocks=true_blocks,
)
print(evals.metrics)
print(evals.confusion) 
```

### Example ground truth for record linkage

```python
data_1 = # your data
data_2 = # your data

true_blocks = pd.DataFrame({
    'x': [0, 1, 2, 3, 4],     # Record indices (reference)
    'y': [3, 1, 4, 0, 2]      # Record indices (Query) 
    'block': [0, 1, 2, 0, 2]  # True block assignments
})

result = blocker.block(
    x=data_1,
    y=data_2,
    true_blocks=true_blocks
)

# Access evaluation metrics
print(result.metrics)    # Shows precision, recall, F1-score, etc.
print(result.confusion)  # Confusion matrix
```
and with `eval` method:

```python
data_1 = # your data
data_2 = # your data

true_blocks = pd.DataFrame({
    'x': [0, 1, 2, 3, 4],    
    'y': [3, 1, 4, 0, 2]     
    'block': [0, 1, 2, 0, 2]  
})

result = blocker.block(
    x=data_1,
    y=data_2,
)
evals = blocker.eval(
    blocking_result=result,
    true_blocks=true_blocks
)
print(evals.metrics) 
print(evals.confusion) 
```

## Choosing an ANN Algorithm

BlockingPy supports multiple ANN algorithms, each with its strengths:

```python
# FAISS (default) - Supports LSH, HNSW and Flat Index
result = blocker.block(x=data, ann="faiss")

# Annoy
result = blocker.block(x=data, ann="annoy")

# HNSW
result = blocker.block(x=data, ann="hnsw")

# Other options: "voyager", "lsh", "kd", "nnd"
```

## Working with lsh or kd algorithm

When the selected algo is lsh or kd, you should specify it in the `control_ann`:

```python
control_ann = {
    "algo" : "lsh",
    "lsh" : {
        # ...
        # your parameters for lsh here
        # ...
    }
}

result = blocker.block(
    x=data,
    ann="lsh",
    control_ann=control_ann,
)
```

## Working with faiss implementation:

When the selected algo is faiss, you should specify which index to use in `control_ann`:

```python
control_ann = {
    "faiss" : {
        "index_type": "flat" or "hnsw" or "lsh"
    }
}
```
