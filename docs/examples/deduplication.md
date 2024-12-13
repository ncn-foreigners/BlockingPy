# Deduplication

This example demonstrates how to use BlockingPy for deduplication of a dataset containing duplicate records. We'll use example data generated with [geco3](https://github.com/T-Strojny/geco3) package which allows for generating data from lookup files or functions and then modifying part of records to create "corrupted" duplicates. This dataset contains 10,000 records, 4,000 of which are duplicates. Original records have 0-2 "corrupted" duplicates and those have 3 modified attributes.

## Setup

First, install BlockingPy:

```python
pip install blockingpy
```

Import required packages:

```python
from blockingpy.blocker import Blocker
import pandas as pd
```

## Data Preparation

Load the example dataset:

```python
data = pd.read_csv('geco_2_dup_per_rec_3_mod.csv')
```

Let's take a look at the data:

```python
data.iloc[50:60, :]

#            rec-id  first_name second_name   last_name              region  \
# 50    rec-038-org      ALICJA    ANTONINA         GIL           POMORSKIE   
# 51    rec-039-org       ZOFIA       HANNA    PAWLICKA        DOLNOŚLĄSKIE   
# 52    rec-040-org      BLANKA       HANNA  WIŚNIEWSKA           LUBELSKIE   
# 53  rec-041-dup-0     NATALIA   KOWALCŹYK  ALEKSANDRA         MAZOWIECKIE   
# 54  rec-041-dup-1         NaN   KOWALCZYK     NATALIA                 NaN   
# 55    rec-041-org  ALEKSANDRA     NATALIA   KOWALCZYK         MAZOWIECKIE   
# 56    rec-042-org       LAURA   MAGDALENA     KONOPKA  KUJAWSKO-POMORSKIE   
# 57    rec-043-org     LILIANA  STANISŁAWA  GRZYBOWSKA        DOLNOŚLĄSKIE   
# 58    rec-044-org     MALWINA       LIDIA    NIEMCZYK           POMORSKIE   
# 59  rec-045-dup-0         NaN     BARBARA        ROSA                 NaN   

#     birth_date personal_id  
# 50  23/01/1953   ZSG686368  
# 51  09/12/1983   MPH633118  
# 52  15/07/1981   SNK483587  
# 53  01/07/1928   MSJ396727  
# 54  01/07/1982   MSJ39682y  
# 55  01/07/1982   MSJ396827  
# 56  03/12/1967   LMH992428  
# 57  16/03/2011   RKG771093  
# 58  24/11/1998   ECJ973778  
# 59  15/07/1960         NaN  
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

# 0    JANAMAŁGORZATAPISAREKMAŁOPOLSKIE25/07/2001SGF898483
# 1                  DETZALEKSANDRAPODKARPACKIETLS812403
# 2    OLIWIAALEKSANDRADECPODKARPACKIE23/04/1944TLS812403
# 3    IRYNAELŻBIETAOSSOWSKAWIELKOPOLSKIE05/12/1950TJD893201
# 4    MATYLDAALEKSANDRAŻUREKZACHODNIOPOMORSKIE28/05/1982LGF327483
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
# ===== starting search (voyager, x, y: 10000,10000, t: 1166) =====
# ===== creating graph =====
```

Let's examine the results:

```python
print(dedup_result)

# ========================================================
# Blocking based on the voyager method.
# Number of blocks: 2833
# Number of columns used for blocking: 1166
# Reduction ratio: 0.9997
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
#          1 | 1812           
#          2 | 600            
#          3 | 210            
#          4 | 87             
#          5 | 42             
#          6 | 30             
#          7 | 16             
#          8 | 16             
#          9 | 9              
#         10 | 3              
#         11 | 2              
#         12 | 2              
#         13 | 2              
#         14 | 1              
#         15 | 1
```
and:

```python
print(dedup_result.result)

#          x     y  block      dist
# 0        1     2      0  0.294986
# 1        9    11      1  0.118932
# 2       15    16      2  0.218879
# 3       18    19      3  0.208658
# 4       31    32      4  0.173967
# ...    ...   ...    ...       ...
# 4827  9992  9993   2828  0.222235
# 4828   643  9994   2829  0.382535
# 4829  8255  9995   2830  0.454295
# 4830  1388  9996   2831  0.384447
# 4831  9998  9999   2832  0.119549
```
Let's take a look at the pair in block `1`:

```python
print(data.iloc[[1,2], : ])

#           rec-id first_name second_name   last_name        region  birth_date  personal_id  
# 1  rec-001-dup-0        NaN        DETZ  ALEKSANDRA  PODKARPACKIE         NaN    TLS812403
# 2    rec-001-org     OLIWIA  ALEKSANDRA         DEC  PODKARPACKIE  23/04/1944    TLS812403
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
print(true_blocks_dedup)

#    x  block
# 0  0      0
# 1  1      1
# 2  2      1
# 3  3      2
# 4  4      3
# 5  5      4
# 6  6      5
# 7  7      6
# 8  8      7
# 9  9      8
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

print(eval_result)

# ========================================================
# Blocking based on the hnsw method.
# Number of blocks: 2850
# Number of columns used for blocking: 1166
# Reduction ratio: 0.9997
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
#          1 | 1707           
#          2 | 681            
#          3 | 233            
#          4 | 96             
#          5 | 55             
#          6 | 32             
#          7 | 16             
#          8 | 11             
#          9 | 9              
#         10 | 3              
#         11 | 3              
#         12 | 1              
#         13 | 1              
#         14 | 1              
#         15 | 1              
# ========================================================
# Evaluation metrics (standard):
# recall : 94.7972
# precision : 24.0766
# fpr : 0.0236
# fnr : 5.2028
# accuracy : 99.976
# specificity : 99.9764
# f1_score : 38.4003
```
The results show:

- High reduction ratio (0.9997) indicating significant reduction in comparison space
- High recall (94.8%) showing most true duplicates are found

The block size distribution shows most blocks contain 1-3 records, with a few larger blocks which could occur due to the fact that even records without duplicates will be grouped it to one of the blocks. This is not a problem since those pairs would not be matched when performing one-to-one record linkage afterwards. This demonstrates BlockingPy's effectiveness at identifying potential duplicates while drastically reducing the number of required comparisons.