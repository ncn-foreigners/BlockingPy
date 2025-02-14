(record_linkage)=
# Record Linkage

This example demonstrates how to use BlockingPy for record linkage between two datasets. We'll use example data created by Paula McLeod, Dick Heasman and Ian Forbes, ONS,
    for the ESSnet DI on-the-job training course, Southampton,
    25-28 January 2011:

- Census: A fictional dataset representing observations from a decennial Census
- CIS: Fictional observations from Customer Information System (combined administrative data from tax and benefit systems)

Some records in the CIS dataset contain Census person IDs, which we'll use to evaluate our blocking performance.

This datasets come with the `BlockingPy` package and can be accesed via `load_census_cis_data` function from `blockingpy.datasets`.

## Setup

First, install BlockingPy:

```bash
pip install blockingpy
```

Import required packages:

```python
from blockingpy import Blocker
from blockingpy.datasets import load_census_cis_data
import pandas as pd
```

## Data Preparation

Download example data:

```python
census, cis = load_census_cis_data()
```

Firstly, we need to filter only those columns which we'll need:

```python
census = census[["PERSON_ID", "PERNAME1", "PERNAME2", "SEX", "DOB_DAY", "DOB_MON", "DOB_YEAR", "ENUMCAP", "ENUMPC"]]
cis = cis[["PERSON_ID", "PERNAME1", "PERNAME2", "SEX", "DOB_DAY", "DOB_MON", "DOB_YEAR", "ENUMCAP", "ENUMPC"]]
```

Let's take a look at the data:

```python
print(census.head())

#       PERSON_ID PERNAME1 PERNAME2 SEX  DOB_DAY  DOB_MON  DOB_YEAR  \
# 0  DE03US001001    COUIE    PRICE   M      1.0        6    1960.0   
# 1  DE03US001002    ABBIE    PVICE   F      9.0       11    1961.0   
# 2  DE03US001003    LACEY    PRICE   F      7.0        2    1999.0   
# 3  DE03US001004   SAMUEL    PRICE   M     13.0        4    1990.0   
# 4  DE03US001005   JOSEPH    PRICE   M     20.0        4    1986.0   

#           ENUMCAP  ENUMPC  
# 0  1 WINDSOR ROAD  DE03US  
# 1  1 WINDSOR ROAD  DE03US  
# 2  1 WINDSOR ROAD  DE03US  
# 3  1 WINDSOR ROAD  DE03US  
# 4  1 WINDSOR ROAD  DE03US

print(cis.head())

#        PERSON_ID  PERNAME1  PERNAME2 SEX  DOB_DAY  DOB_MON  DOB_YEAR  \
# 0  PO827ER091001    HAYDEN      HALL   M      NaN        1       NaN   
# 1  LS992DB024001     SEREN  ANDERSON   F      1.0        1       NaN   
# 2   M432ZZ053003     LEWIS     LEWIS   M      1.0        1       NaN   
# 3   SW75TQ018001  HARRISON    POSTER   M      5.0        1       NaN   
# 4  EX527TR017006  MUHAMMED    WATSUN   M      7.0        1       NaN   

#               ENUMCAP   ENUMPC  
# 0    91 CLARENCE ROAD  PO827ER  
# 1      24 CHURCH LANE  LS992DB  
# 2      53 CHURCH ROAD   M432ZZ  
# 3   19 HIGHFIELD ROAD   SW75TG  
# 4  17 VICTORIA STREET      NaN  

print(census.shape)
# (25343, 9)

print(cis.shape)
# (24613, 9)
```

Preprocess data and create column `txt` containing concatenated variables:

```python
# Convert numeric fields to strings
census[['DOB_DAY', 'DOB_MON', 'DOB_YEAR']] = census[['DOB_DAY', 'DOB_MON', 'DOB_YEAR']].astype(str)
cis[['DOB_DAY', 'DOB_MON', 'DOB_YEAR']] = cis[['DOB_DAY', 'DOB_MON', 'DOB_YEAR']].astype(str)

# Fill NAs with empty strings
census = census.fillna('')
cis = cis.fillna('')

# Concatenate fields
census['txt'] = census['PERNAME1'] + census['PERNAME2'] + census['SEX'] + \
                census['DOB_DAY'] + census['DOB_MON'] + census['DOB_YEAR'] + \
                census['ENUMCAP'] + census['ENUMPC']

cis['txt'] = cis['PERNAME1'] + cis['PERNAME2'] + cis['SEX'] + \
             cis['DOB_DAY'] + cis['DOB_MON'] + cis['DOB_YEAR'] + \
             cis['ENUMCAP'] + cis['ENUMPC']
```

Let's see how the new column looks like:

```python
print(census['txt'].head())

# txt
# 0	COUIEPRICEM1.061960.01 WINDSOR ROADDE03US
# 1	ABBIEPVICEF9.0111961.01 WINDSOR ROADDE03US
# 2	LACEYPRICEF7.021999.01 WINDSOR ROADDE03US
# 3	SAMUELPRICEM13.041990.01 WINDSOR ROADDE03US
# 4	JOSEPHPRICEM20.041986.01 WINDSOR ROADDE03US

print(cis['txt'].head())

# 	txt
# 0	HAYDENHALLMnan1nan91 CLARENCE ROADPO827ER
# 1	SERENANDERSONF1.01nan24 CHURCH LANELS992DB
# 2	LEWISLEWISM1.01nan53 CHURCH ROADM432ZZ
# 3	HARRISONPOSTERM5.01nan19 HIGHFIELD ROADSW75TG
# 4	MUHAMMEDWATSUNM7.01nan17 VICTORIA STREET

```

## Perform record linkage

Initialize blocker instance and perform blocking with `hnsw` algorithm, `cosine` distance and custom parameters:

```python
blocker = Blocker()

control_ann = {
    "hnsw": {
        'distance': "cosine",
        'M': 40,
        'ef_c': 500,
        'ef_s': 500
    }
}

rec_lin_result = blocker.block(
    x=census['txt'],
    y=cis['txt'],   
    ann='hnsw',    
    verbose=1,      
    control_ann=control_ann, 
    # control_txt=control_txt, # let's leave this as default
)

# Output:

# ===== creating tokens =====
# ===== starting search (hnsw, x, y: 25343,24613, t: 1072) =====
# ===== creating graph =====
```

Let's take a look at the results:

```python
print(rec_lin_result)

# ========================================================
# Blocking based on the hnsw method.
# Number of blocks: 23996
# Number of columns used for blocking: 1072
# Reduction ratio: 0.999961
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
#          2 | 23392          
#          3 | 592            
#          4 | 11             
#          5 | 1   

print(rec_lin_result.result.head())

#            x      y  block      dist
# 0      17339      0      0  0.273628
# 1       9567      1      1  0.103388
# 2      10389      2      2  0.107852
# 3      24258      3      3  0.211039
# 4       3714      4      4  0.294986
```
Let's take a look at the pair in block `0` :
```python
print(cis.iloc[0, :])
print(census.iloc[17339, :])

# PERSON_ID                                PO827ER091001
# PERNAME1                                        HAYDEN
# PERNAME2                                          HALL
# SEX                                                  M
# DOB_DAY                                            nan
# DOB_MON                                              1
# DOB_YEAR                                           nan
# ENUMCAP                               91 CLARENCE ROAD
# ENUMPC                                         PO827ER
# txt          HAYDENHALLMnan1nan91 CLARENCE ROADPO827ER
# Name: 0, dtype: object
# PERSON_ID                                   PO827ER091001
# PERNAME1                                           HAYDEM
# PERNAME2                                             HALL
# SEX                                                     M
# DOB_DAY                                               1.0
# DOB_MON                                                 1
# DOB_YEAR                                           1957.0
# ENUMCAP                                  91 CLARENCE ROAD
# ENUMPC                                            PO827ER
# txt          HAYDEMHALLM1.011957.091 CLARENCE ROADPO827ER
# Name: 17339, dtype: object

```

## Evaluate Results

Firstly, we need to prepare `true_blocks` DataFrame from our data (using known `person_id` in both datasets):

```python
# Create x and y indices
census['x'] = range(len(census))
cis['y'] = range(len(cis))

# Find true matches using person_id
matches = pd.merge(
    left=census[['PERSON_ID', 'x']],
    right=cis[['PERSON_ID', 'y']],
    on='PERSON_ID'
)

# Add block numbers
matches['block'] = range(len(matches))

matches.shape
# (24043, 4)
```
Let's sample 1000 pairs for which we will evaluate:
```python
matches = matches.sample(1000, random_state=42)
```

Now we can evaluate the algorithm:

```python
# Perform blocking with evaluation
eval_result = blocker.block(
    x=census['txt'],
    y=cis['txt'],
    true_blocks=matches[['x', 'y', 'block']],
    verbose=1,
    ann='faiss'  # Try a different algorithm
)

# ===== creating tokens =====
# ===== starting search (hnsw, x, y: 25343,24613, t: 1072) =====
# ===== creating graph =====

# alternatively we can use the `eval` method for separation:
# result = blocker.block(
#     x=census['txt'],
#     y=cis['txt'],
#     verbose=1,
#     ann='faiss'
# )
# eval_result = blocker.eval(
#     blocking_result=result,
#     true_blocks=matches[['x', 'y', 'block']]
#)
# The procedure in both cases stays the same.

# Note: We recommend using eval() method when evaluating larger datasets 
# since it allows you to set the batch size for currently evaluated record pairs.
```

and print results with evaluation metrics:

```python
print(eval_result)
# ========================================================
# Blocking based on the faiss method.
# Number of blocks: 23984
# Number of columns used for blocking: 1072
# Reduction ratio: 1.0000
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
#          2 | 23369          
#          3 | 602            
#          4 | 12             
#          5 | 1                      
# ========================================================
print(eval_result.metrics)
# Evaluation metrics (standard):
# recall : 99.691
# precision : 99.691
# fpr : 0.0
# fnr : 0.309
# accuracy : 100.0
# specificity : 100.0
# f1_score : 99.691
```
The output shows:

- Reduction ratio (how much the comparison space was reduced)
- Block size distribution

If true matches were provided:

- Recall
- Precision
- False positive rate
- False negative rate
- Accuracy
- Specificity 
- F1 score



For this example, using `faiss` we achieve:

- 99.69% recall and precision
- close to 100% accuracy
- Near perfect reduction ratio of 1.0
- Most blocks contain just 2-3 records

This demonstrates BlockingPy's effectiveness at finding matching records while drastically reducing the number of required comparisons.