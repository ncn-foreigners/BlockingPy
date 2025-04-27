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

# 	fname_c1	fname_c2	lname_c1	lname_c2   by	bm	bd	id  true_id
# 0	FRANK	    NaN	        MUELLER	    NaN	       1967	9	27	1	3606
# 1	MARTIN	    NaN	        SCHWARZ	    NaN	       1967	2	17	2	2560
# 2	HERBERT	    NaN	        ZIMMERMANN  NaN	       1961	11	6	3	3892
# 3	HANS	    NaN	        SCHMITT	    NaN	       1945	8	14	4	329
# 4	UWE	    NaN	        KELLER	    NaN	       2000	7	5	5	1994
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

# 0         FRANKMUELLER1967927
# 1        MARTINSCHWARZ1967217
# 2    HERBERTZIMMERMANN1961116
# 3          HANSSCHMITT1945814
# 4             UWEKELLER200075
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
```

We can now take a look at the results: 
```python
print(dedup_result)

# ========================================================
# Blocking based on the hnsw method.
# Number of blocks: 2425
# Number of columns used for blocking: 512
# Reduction ratio: 0.999505
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
#          2 | 796            
#          3 | 548            
#          4 | 328            
#          5 | 251            
#          6 | 153            
#          7 | 108            
#          8 | 70             
#          9 | 54             
#         10 | 34             
#         11 | 28             
#         12 | 15             
#         13 | 8              
#         14 | 9              
#         15 | 8              
#         16 | 4              
#         17 | 4              
#         19 | 1              
#         20 | 1              
#         22 | 1              
#         23 | 2              
#         28 | 1              
#         32 | 1  
```

and:

```python
print(dedup_result.result)
#          x     y  block      dist
# 0     2337     0      0  0.170293
# 1     8081     1      1  0.213184
# 2     4712     2      2  0.163775
# 3     1956     3      3  0.158502
# 4     4931     4      4  0.279722
# ...    ...   ...    ...       ...
# 7570  4139  9991    207  0.261605
# 7571  1230  9994    365  0.295026
# 7572  6309  9995     84  0.163340
# 7573  7553  9996   1695  0.130726
# 7574  5396  9997     62  0.295574
```

Let's see the pair in the `block` no. `3`

```python
print(data.iloc[[1956, 3], : ])
#      fname_c1 fname_c2 lname_c1  ...    id true_id                  txt
# 1956    HRANS           SCHMITT  ...  1957     329  HRANSSCHMITT1945814
# 3        HANS           SCHMITT  ...     4     329   HANSSCHMITT1945814
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
# 0    FRANK              MUELLER  ...       FRANKMUELLER1967927  3606  0
# 1   MARTIN              SCHWARZ  ...      MARTINSCHWARZ1967217  2560  1
# 2  HERBERT           ZIMMERMANN  ...  HERBERTZIMMERMANN1961116  3892  2
# 3     HANS              SCHMITT  ...        HANSSCHMITT1945814   329  3
# 4      UWE               KELLER  ...           UWEKELLER200075  1994  4
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
```

You can also inspect:

```python
print(eval_result.metrics)
# recall         0.963000
# precision      0.038751
# fpr            0.000478
# fnr            0.037000
# accuracy       0.999521
# specificity    0.999522
# f1_score       0.074504
# dtype: float64
print(eval_result.confusion)
#                  Predicted Positive  Predicted Negative
# Actual Positive                 963                  37
# Actual Negative               23888            49970112
```

## Summary
Comparing both methods, we can see that using embeddings performed slightly worse than the traditional shingle-based approach in this example (`96.3%` recall vs. `100%` with shingles).
However, embeddings still provide a viable and effective solution for deduplication.
In certain datasets or conditions embeddings may even outperform  shingle-based methods.