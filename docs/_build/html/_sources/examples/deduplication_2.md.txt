# Deduplication No. 2

In this example we'll use data known as `RLdata10000` taken from [RecordLinkage](https://cran.r-project.org/package=RecordLinkage) R package developed by Murat Sariyar
and Andreas Borg. It contains 10 000 records in total where some have been duplicated with randomly generated errors. There are 9000 original records and 1000 duplicates.

## Data Preparation

Let's install `blockingpy`

```bash
pip install blockingpy
```

Import necessary packages and functions:

```python
import pandas as pd
from blockingpy import Blocker
from blockingpy.datasets import load_deduplication_data()
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

Let's perfrom basic deduplication using `hnsw` algorithm

```python
blocker = Blocker()
dedup_result = blocker.block(
    x=data['txt'],
    ann='hnsw',
    verbose=1,
)

# ===== creating tokens =====
# ===== starting search (hnsw, x, y: 10000,10000, t: 674) =====
# ===== creating graph =====
```

We can now take a look at the results: 

```python
print(dedup_result)

# ========================================================
# Blocking based on the hnsw method.
# Number of blocks: 2736
# Number of columns used for blocking: 674
# Reduction ratio: 0.9996
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
#          2 | 962            
#          3 | 725            
#          4 | 409            
#          5 | 263            
#          6 | 139            
#          7 | 89             
#          8 | 52             
#          9 | 37             
#         10 | 24             
#         11 | 14             
#         12 | 9              
#         13 | 5              
#         14 | 2              
#         15 | 1              
#         16 | 1              
#         17 | 2              
#         20 | 1              
#         64 | 1   
```

and:

```python
print(dedup_result.result)
#          x     y  block      dist
# 0     3402     0      0  0.256839
# 1     1179     1      1  0.331352
# 2     2457     2      2  0.209737
# 3     1956     3      3  0.085341
# 4     4448     4      4  0.375000
# ...    ...   ...    ...       ...
# 7259  9206  9994   1981  0.390912
# 7260  6309  9995   1899  0.268436
# 7261  5162  9996   1742  0.188893
# 7262  6501  9997   1293  0.245406
# 7263  5183  9999   1273  0.209088
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

Now we can evaluate our algorithm:

```python
control_ann = {
    "faiss":{
        "distance": "cosine"
    }
}

blocker = Blocker()
eval_result = blocker.block(
    x=df_eval['txt'], 
    ann='faiss',
    true_blocks=true_blocks_dedup, 
    verbose=1, 
    control_ann=control_ann
)
# ===== creating tokens =====
# ===== starting search (faiss, x, y: 10000,10000, t: 674) =====
# ===== creating graph =====
```
And the results:

```python
print(eval_result)
print(eval_result.metrics)
# ========================================================
# Blocking based on the faiss method.
# Number of blocks: 2737
# Number of columns used for blocking: 674
# Reduction ratio: 0.9996
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
#          2 | 972            
#          3 | 721            
#          4 | 423            
#          5 | 236            
#          6 | 138            
#          7 | 92             
#          8 | 62             
#          9 | 29             
#         10 | 28             
#         11 | 15             
#         12 | 8              
#         13 | 3              
#         14 | 3              
#         15 | 1              
#         16 | 1              
#         17 | 2              
#         18 | 1              
#         20 | 1              
#         67 | 1              
# ========================================================
# Evaluation metrics (standard):
# recall : 100.0
# precision : 4.7651
# fpr : 0.04
# fnr : 0.0
# accuracy : 99.96
# specificity : 99.96
# f1_score : 9.0967
# recall         1.000000
# precision      0.047651
# fpr            0.000400
# fnr            0.000000
# accuracy       0.999600
# specificity    0.999600
# f1_score       0.090967
```

```python
print(eval_result.confusion)
# 	                Actual Negative     Actual Positive
# Predicted Negative	49974014	    0
# Predicted Positive	19986	            1000
```

The results show high reduction ratio `0.9996` alongside perfect recall (`1.000`) indicating that our package handled this dataset very well.