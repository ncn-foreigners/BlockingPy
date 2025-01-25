# Deduplication

This example demonstrates how to use BlockingPy for deduplication of a dataset containing duplicate records. We'll use example data generated with [geco3](https://github.com/T-Strojny/geco3) package which allows for generating data from lookup files or functions and then modifying part of records to create "corrupted" duplicates. This dataset contains 10,000 records, 4,000 of which are duplicates. Original records have 0-2 "corrupted" duplicates and those have 3 modified attributes.

## Setup

First, install BlockingPy:

```python
pip install blockingpy
```

Import required packages:

```python
from blockingpy import Blocker
import pandas as pd
```

## Data Preparation

Load the example dataset:

```python
data = pd.read_csv('geco_2_dup_per_rec_3_mod.csv')
```

Let's take a look at the data:

```python
data.iloc[40:50, :]

#            rec-id  first_name second_name   last_name              region  \
# 40    rec-024-org        MAJA        OLGA     LEWICKA  ZACHODNIOPOMORSKIE   
# 41    rec-025-org        POLA    LEOKADIA   RUTKOWSKA  ZACHODNIOPOMORSKIE   
# 42  rec-026-dup-0  ALEKSANDRA       RYBAK       ZÓFIA  KUJAWSKO-POMORSKIE   
# 43  rec-026-dup-1  ALEKSANDRA       RYBAK       ZÓFIA  KUJAWSKO-POMORSKIE   
# 44    rec-026-org       ZOFIA  ALEKSANDRA       RYBAK  KUJAWSKO-POMORSKIE   
# 45  rec-027-dup-0       LAÓRA    JAGYEŁŁO      JOANNA       WIELKOPOLSKIE   
# 46    rec-027-org       LAURA      JOANNA    JAGIEŁŁO       WIELKOPOLSKIE   
# 47  rec-028-dup-0       MARIA        KOZA    WIKTÓRIA        DOLNOŚLĄSKIE   
# 48    rec-028-org    WIKTORIA       MARIA        KOZA        DOLNOŚLĄSKIE   
# 49    rec-029-org      NIKOLA  BRONISŁAWA  WIĘCKOWSKA             ŚLĄSKIE   

#     birth_date personal_id  
# 40  22/10/1935   DKK423341  
# 41  29/11/1956   LJL907920  
# 42         NaN   DAT77p499  
# 43         NaN         NaN  
# 44  24/03/1982   DAT770499  
# 45  10/11/1984   LNRt57399  
# 46  10/11/1984   LNR657399  
# 47         NaN   HEH671979  
# 48  09/09/1982   HEH671989  
# 49  09/11/1992   JKR103426  
```

Preprocess data by concatenating all fields into a single text column:

```python
data['txt'] = (
    data['first_name'].fillna('') +
    data['second_name'].fillna('') +
    data['last_name'].fillna('') + 
    data['region'].fillna('') +
    data['birth_date'].fillna('') +
    data['personal_id'].fillna('')
)

print(data['txt'].head())

# 0	GÓRKAKARÓLINAMELANIIAŚWIĘTOKRZYSKIE25/07/2010S...
# 1	MELANIAKAROLINAGÓRKAŚWIĘTOKRZYSKIE25/07/2001SG...
# 2	MARTAMARTYNAMUSIAŁPODKARPACKIE23/04/1944TLS812403
# 3	KAJAPATRYCJADROZDDOLNOŚLĄSKIE05/12/1950TJH243280
# 4	HANNAKLARALIPSKAMAŁOPOLSKIE28/05/1991MTN763673
```

## Basic Deduplication

Initialize blocker instance and perform deduplication using the Voyager algorithm:

```python
control_ann = {
    'voyager': {
        'distance': 'cosine',
        'random_seed': 42,
        'M': 16,
        'ef_construction': 300,
    }
}

blocker = Blocker()
dedup_result = blocker.block(
    x=data['txt'],
    ann='voyager',
    verbose=1,
    control_ann=control_ann
)

# ===== creating tokens =====
# ===== starting search (voyager, x, y: 10000,10000, t: 1169) =====
# ===== creating graph =====
```

Let's examine the results:

```python
print(dedup_result)

# ========================================================
# Blocking based on the voyager method.
# Number of blocks: 2711
# Number of columns used for blocking: 1169
# Reduction ratio: 0.9996
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
#          2 | 917            
#          3 | 877            
#          4 | 367            
#          5 | 204            
#          6 | 105            
#          7 | 77             
#          8 | 34             
#          9 | 26             
#         10 | 30             
#         11 | 12             
#         12 | 10             
#         13 | 13             
#         14 | 12             
#         15 | 10             
#         16 | 4              
#         17 | 3              
#         18 | 2              
#         19 | 2              
#         21 | 1              
#         22 | 3              
#         27 | 1              
#         30 | 1   
```
and:

```python
print(dedup_result.result)

#          x     y  block      dist
# 0        1     0      0  0.102041
# 1     5974     2      1  0.390295
# 2     7378     3      2  0.425410
# 3     5562     4      3  0.396494
# 4     1389     5      4  0.461184
# ...    ...   ...    ...       ...
# 7298  9995  9993   2710  0.241895
# 7299  9995  9994   2710  0.135667
# 7300  4029  9996   1549  0.386845
# 7301  9999  9997     66  0.128579
# 7302  9999  9998     66  0.140395
```
Let's take a look at the pair in block `1`:

```python
print(data.iloc[[7,8], : ])

#              rec-id first_name second_name   last_name               region        birth_date personal_id                                                       txt
# 9998  rec-999-dup-1     RESŻKA    LILIANNA  MAŁGÓRZATA  WARMIŃSKO-MAZURSKIE         12/01/1978        NaN         RESŻKALILIANNAMAŁGÓRZATAWARMIŃSKO-MAZURSKIE12/...
# 9999    rec-999-org   LILIANNA  MAŁGORZATA      RESZKA  WARMIŃSKO-MAZURSKIE         12/01/1978   TCX847483        LILIANNAMAŁGORZATARESZKAWARMIŃSKO-MAZURSKIE12/...
```
Even though records differ a lot, our package managed to get this pair right.

## Evaluation with True Blocks

Since our dataset contains known duplicate information in the `rec-id` field, we can evaluate the blocking performance. First, we'll prepare the true blocks information:

```python
df_eval = data.copy()

# Extract block numbers from rec-id
df_eval['block'] = df_eval['rec-id'].str.extract(r'rec-(\d+)-')
df_eval['block'] = df_eval['block'].astype('int')

# Add sequential index
df_eval = df_eval.sort_values(by=['block'], axis=0).reset_index()
df_eval['x'] = range(len(df_eval))

# Prepare true blocks dataframe
true_blocks_dedup = df_eval[['x', 'block']]
```
Print `true_blocks_dedup`:

```python
print(true_blocks_dedup.head(10))

#    x  block
# 0  0      0
# 1  1      0
# 2  2      1
# 3  3      2
# 4  4      3
# 5  5      4
# 6  6      5
# 7  7      6
# 8  8      6
# 9  9      7
```

Now we can perform blocking with evaluation using the HNSW algorithm:

```python
control_ann = {
    "hnsw": {
        'distance': "cosine",
        'M': 40,
        'ef_c': 500,
        'ef_s': 500
    }
}

blocker = Blocker()
eval_result = blocker.block(
    x=df_eval['txt'], 
    ann='hnsw',
    true_blocks=true_blocks_dedup, 
    verbose=1, 
    control_ann=control_ann
)
# We can also evaluate separately with `eval` method:
# result = blocker.block(
#     x=df_eval['txt'], 
#     ann='hnsw', 
#     verbose=1, 
#     control_ann=control_ann
# )
# eval_result = blocker.eval(
#     blocking_result=result,
#     true_blocks=true_blocs_dedup
# ) 
# The rest stays the same in both cases
#Note: We recommend using eval() method when evaluating larger datasets since it allows you to set the batch #size for currently evaluated record pairs.

print(eval_result)
print(eval_result.metrics)
# ========================================================
# Blocking based on the hnsw method.
# Number of blocks: 2972
# Number of columns used for blocking: 1169
# Reduction ratio: 0.9996
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
#          2 | 1113           
#          3 | 997            
#          4 | 391            
#          5 | 200            
#          6 | 88             
#          7 | 65             
#          8 | 39             
#          9 | 19             
#         10 | 16             
#         11 | 13             
#         12 | 9              
#         13 | 8              
#         14 | 4              
#         15 | 1              
#         16 | 3              
#         17 | 1              
#         18 | 2              
#         19 | 1              
#         22 | 1              
#         25 | 1              
# ========================================================
# Evaluation metrics (standard):
# recall : 99.0151
# precision : 29.2353
# fpr : 0.0248
# fnr : 0.9849
# accuracy : 99.9751
# specificity : 99.9752
# f1_score : 45.142
```
The results show:

- High reduction ratio (`0.9996`) indicating significant reduction in comparison space
- High recall (`99.02%`) showing most true duplicates are found

The block size distribution shows most blocks contain 1-3 records, with a few larger blocks which could occur due to the fact that even records without duplicates will be grouped it to one of the blocks. This is not a problem since those pairs would not be matched when performing one-to-one comparison afterwards. This demonstrates BlockingPy's effectiveness at identifying potential duplicates while drastically reducing the number of required comparisons.