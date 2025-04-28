(quickstart)=
# Quick Start

This guide will help you get started with BlockingPy by walking through some basic examples. We'll cover both record linkage (matching records between two datasets) and deduplication (finding duplicates within a single dataset).

## Basic Record Linkage

Let's start with a simple example of matching records between two datasets. We'll use names that have slight variations to demonstrate how BlockingPy handles approximate matching.

Firstly, we will import our main blocker class `Blocker` used for blocking from BlockingPy and Pandas:

```python
from blockingpy import Blocker
import pandas as pd
```

Now let's create simple datasets for our example:

```python
dataset1 = pd.DataFrame({
    "txt": [
        "johnsmith",
        "smithjohn",
        "smiithhjohn",
        "smithjohnny",
        "montypython",
        "pythonmonty",
        "errmontypython",
        "monty",
    ]
})

dataset2 = pd.DataFrame({
    "txt": [
        "montypython",
        "smithjohn",
        "other",
    ]
})
```
We initialize the `Blocker` instance and perform blocking:

```python
blocker = Blocker()
blocking_result = blocker.block(x=dataset1['txt'], y=dataset2['txt'])
```
Let's print `blocking_result` and see the output:

```python
print(blocking_result)
# ========================================================
# Blocking based on the faiss method.
# Number of blocks: 3
# Number of columns used for blocking: 17
# Reduction ratio: 0.8750
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
#          2 | 3
```
Our output contains:

- Algorithm used for blocking (default - `faiss - HNSW index`)
- Number of blocks created
- Number of columns used for blocking (obtained by creating DTMs from datasets)
- Reduction ratio i.e. how large is the reduction of comparison pairs (here `0.8750` which means blocking reduces comparison by over 87.5%).

We can print `blocking_result.result` to get the detailed matching results:

```python
print(blocking_result.result)
#    x  y  block  dist
# 0  4  0      0   0.0
# 1  1  1      1   0.0
# 2  7  2      2   6.0
```

Here we have:

- `x`: Index from the first dataset (dataset1)
- `y`: Index from the second dataset (dataset2)
- `block`: The block ID these records were grouped into
- `dist`: The distance between the records (smaller means more similar)

## Basic Deduplication

Now let's try finding duplicates within a single dataset:

```python
dedup_result = blocker.block(x=dataset1['txt'])

print(dedup_result)
# ========================================================
# Blocking based on the faiss method.
# Number of blocks: 2
# Number of columns created for blocking: 25
# Reduction ratio: 0.5714
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
#          4 | 2
```
Output contains similar information as the record linkage one:

- `faiss` algorithm used
- `2` blocks created
- `25` columns (features) created for blocking from text representation
- `0.5714` reduction ratio (meaning we get about `57.14%` reduction in comparison pairs)

Let's take a look at the detailed information:
```python
print(dedup_result.result)
#    x  y  block  dist
# 0  0  1      0   2.0
# 1  1  2      0   2.0
# 2  1  3      0   2.0
# 3  4  5      1   2.0
# 4  4  6      1   3.0
# 5  4  7      1   6.0
```

## Understanding the Results

BlockingPy uses character n-grams and approximate nearest neighbor algorithms to group similar records together. By default, it uses the FAISS algorithm with sensible default parameters.

The reduction ratio shows how much the blocking reduces the number of required comparisons. For example, a ratio of `0.8750` means the blocking eliminates 87.5% of possible comparisons, greatly improving efficiency while maintaining accuracy.

## Next Steps

This quick start covered the basics using default settings. BlockingPy offers several additional features:

- Multiple ANN algorithms (Faiss, HNSW, Voyager, Annoy, MLPack, NND)
- Various distance metrics
- Custom text processing options (Embeddings or Ngrams)
- Performance tuning parameters
- Eval metrics when true blocks are known

Check out the {ref}`user-guide` for more detailed examples and configuration options.