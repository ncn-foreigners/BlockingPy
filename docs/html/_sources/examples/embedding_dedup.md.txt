# Deduplication with Embeddings

This tutorial demonstrates how to use the `BlockingPy` library for deduplication using embeddings instead of n-gram shingles. It is based on the [Deduplication No. 2 tutorial](https://blockingpy.readthedocs.io/en/latest/examples/deduplication_2.html), but adapted to showcase the use of embeddings.

Once again, we will use the  `RLdata10000` dataset taken from [RecordLinkage](https://cran.r-project.org/package=RecordLinkage) R package developed by Murat Sariyar
and Andreas Borg. It contains 10 000 records in total where some have been duplicated with randomly generated errors. There are 9000 original records and 1000 duplicates.

## Data Preparation

Let's install `blockingpy`:

```bash
pip install blockingpy
```

Import necessary packages and functions:

```python
import pandas as pd
from blockingpy import Blocker
from blockingpy.datasets import load_deduplication_data
```

Let's load the data and take a look at first 5 rows:

```python
data = load_deduplication_data()
data.head()

#   fname_c1 fname_c2    lname_c1 lname_c2    by  bm  bd  id  true_id
# 0    FRANK      NaN     MUELLER      NaN  1967   9  27   1     3606
# 1   MARTIN      NaN     SCHWARZ      NaN  1967   2  17   2     2560
# 2  HERBERT      NaN  ZIMMERMANN      NaN  1961  11   6   3     3892
# 3     HANS      NaN     SCHMITT      NaN  1945   8  14   4      329
# 4      UWE      NaN      KELLER      NaN  2000   7   5   5     1994
```

Now we need to prepare the `txt` column:

```python
data = data.fillna('')
data[['by', 'bm', 'bd']] = data[['by', 'bm', 'bd']].astype('str')
data['txt'] = (
    data["fname_c1"] +
    data["fname_c2"] +
    data['lname_c1'] +
    data['lname_c2'] +
    data['by'] +
    data['bm'] +
    data['bd']
    )   
data['txt'].head()

# 0         FRANK  MUELLER  1967 9 27
# 1        MARTIN  SCHWARZ  1967 2 17
# 2    HERBERT  ZIMMERMANN  1961 11 6
# 3          HANS  SCHMITT  1945 8 14
# 4             UWE  KELLER  2000 7 5
# Name: txt, dtype: object
```

## Basic Deduplication

We'll now perform basic deduplication with `hnsw` algorithm, but instead of character-level n-grams, the text will be encoded into dense embeddings before approximate nearest neighbor search.

```python
blocker = Blocker()

control_txt = {
    "encoder": "embedding",
    "embedding": {
        "model": "minishlab/potion-base-32M",
        # for other customization options see 
        # configuration in User Guide
    }
}

dedup_result = blocker.block(
    x=data['txt'],
    ann='hnsw',
    verbose=1,
    random_seed=42,
    control_txt=control_txt,
)
# ===== creating tokens: embedding =====
# ===== starting search (hnsw, x, y: 10000,10000, t: 512) =====
# ===== creating graph =====
```

We can now take a look at the results: 
```python
print(dedup_result)

# ========================================================
# Blocking based on the hnsw method.
# Number of blocks: 2656
# Number of columns created for blocking: 512
# Reduction ratio: 0.999600
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
#          2 | 906            
#          3 | 631            
#          4 | 424            
#          5 | 273            
#          6 | 155            
#          7 | 121            
#          8 | 48             
#          9 | 34             
#         10 | 23             
#         11 | 14             
#         12 | 13             
#         13 | 4              
#         14 | 3              
#         15 | 2              
#         16 | 1              
#         18 | 1              
#         20 | 1              
#         23 | 1              
#         26 | 1     
```

and:

```python
print(dedup_result.result)
#          x     y  block      dist
# 0     2337     0      0  0.227015
# 1     4504     1      1  0.373196
# 2      233     2      2  0.294851
# 3     1956     3      3  0.261316
# 4     4040     4      4  0.216883
# ...    ...   ...    ...       ...
# 7339  6692  9984   2328  0.338963
# 7340  5725  9986   1532  0.243514
# 7341  8521  9993   1915  0.324314
# 7342  7312  9997    774  0.235769
# 7343  5897  9999   1558  0.217153
```

Let's see the pair in the `block` no. `3`

```python
print(data.iloc[[1956, 3], : ])
#      fname_c1 fname_c2 lname_c1  ...    id true_id                      txt
# 1956    HRANS           SCHMITT  ...  1957     329  HRANS SCHMITT 1945 8 14
# 3        HANS           SCHMITT  ...     4     329   HANS SCHMITT 1945 8 14
```

## True Blocks Preparation

```python
df_eval = data.copy()
df_eval['block'] = df_eval['true_id']
df_eval['x'] = range(len(df_eval))
```

```python
print(df_eval.head())
#   fname_c1 fname_c2    lname_c1  ...                       txt block  x
# 0    FRANK              MUELLER  ...       FRANK MUELLER 1967 9  27  3606  0
# 1   MARTIN              SCHWARZ  ...      MARTIN SCHWARZ 1967 2 17  2560  1
# 2  HERBERT           ZIMMERMANN  ...  HERBERT ZIMMERMANN 1961 1 16  3892  2
# 3     HANS              SCHMITT  ...        HANS SCHMITT 1945 8 14   329  3
# 4      UWE               KELLER  ...           UWE KELLER 2000 7 5  1994  4
```

Let's create the final `true_blocks_dedup`:

```python
true_blocks_dedup = df_eval[['x', 'block']]
```
## Evaluation

Finally, we can evaluate the blocking performance when using embeddings:

```python
blocker = Blocker()
eval_result = blocker.block(
    x=df_eval['txt'], 
    ann='voyager',
    true_blocks=true_blocks_dedup, 
    verbose=1, 
    random_seed=42,
    control_txt=control_txt, # Using the same config
)
# ===== creating tokens: embedding =====
# ===== starting search (voyager, x, y: 10000,10000, t: 512) =====
# ===== creating graph =====
# ===== evaluating =====
```

You can also inspect:

```python
print(eval_result.metrics)
# recall         0.957000
# precision      0.047266
# fpr            0.000386
# fnr            0.043000
# accuracy       0.999613
# specificity    0.999614
# f1_score       0.090083
# dtype: float64
print(eval_result.confusion)
#                  Predicted Positive  Predicted Negative
# Actual Positive                 957                  43
# Actual Negative               19290            49974710
```

## Summary
Comparing both methods, we can see that using embeddings performed slightly worse than the traditional shingle-based approach in this example (`95.7%` recall vs. `100%` with shingles).
However, embeddings still provide a viable and effective solution for deduplication.
In certain datasets or conditions embeddings may even outperform  shingle-based methods.