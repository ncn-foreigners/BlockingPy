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
census[['DOB_DAY', 'DOB_MON', 'DOB_YEAR']] = census[['DOB_DAY', 'DOB_MON', 'DOB_YEAR']].astype("Int64").astype(str).replace('<NA>', '')
cis[['DOB_DAY', 'DOB_MON', 'DOB_YEAR']] = cis[['DOB_DAY', 'DOB_MON', 'DOB_YEAR']].astype("Int64").astype(str).replace('<NA>', '')

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
# 0      COUIEPRICEM1619601 WINDSOR ROADDE03US
# 1     ABBIEPVICEF91119611 WINDSOR ROADDE03US
# 2      LACEYPRICEF7219991 WINDSOR ROADDE03US
# 3    SAMUELPRICEM13419901 WINDSOR ROADDE03US
# 4    JOSEPHPRICEM20419861 WINDSOR ROADDE03US

print(cis['txt'].head())

# 	txt
# 0         HAYDENHALLM191 CLARENCE ROADPO827ER
# 1       SERENANDERSONF1124 CHURCH LANELS992DB
# 2           LEWISLEWISM1153 CHURCH ROADM432ZZ
# 3    HARRISONPOSTERM5119 HIGHFIELD ROADSW75TG
# 4         MUHAMMEDWATSUNM7117 VICTORIA STREET

```

## Perform record linkage

Initialize blocker instance and perform blocking with `hnsw` algorithm and default parameters:

```python
blocker = Blocker()

rec_lin_result = blocker.block(
    x=census['txt'],
    y=cis['txt'],   
    ann='hnsw',    
    verbose=1,      
    random_seed=42
)

# Output:

# ===== creating tokens: shingle =====
# ===== starting search (hnsw, x, y: 25343,24613, t: 1072) =====
# ===== creating graph =====
```

Let's take a look at the results:

```python
print(rec_lin_result)

# ========================================================
# Blocking based on the hnsw method.
# Number of blocks: 23993
# Number of columns created for blocking: 1072
# Reduction ratio: 0.999961
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
        #  2 | 23388          
        #  3 | 591            
        #  4 | 13             
        #  5 | 1    

print(rec_lin_result.result.head())

#      x      y  block      dist
# 0    17339  0      0  0.134151
# 1    9567   1      1  0.064307
# 2    10389  2      2  0.044183
# 3    24258  3      3  0.182125
# 4    3714   4      4  0.288487
```
Let's take a look at the pair in block `0` :
```python
print(cis.iloc[0, :])
print(census.iloc[17339, :])

# PERSON_ID                          PO827ER091001
# PERNAME1                                  HAYDEN
# PERNAME2                                    HALL
# SEX                                            M
# DOB_DAY                                         
# DOB_MON                                        1
# DOB_YEAR                                        
# ENUMCAP                         91 CLARENCE ROAD
# ENUMPC                                   PO827ER
# txt          HAYDENHALLM191 CLARENCE ROADPO827ER
# Name: 0, dtype: object

# PERSON_ID                               PO827ER091001
# PERNAME1                                       HAYDEM
# PERNAME2                                         HALL
# SEX                                                 M
# DOB_DAY                                             1
# DOB_MON                                             1
# DOB_YEAR                                         1957
# ENUMCAP                              91 CLARENCE ROAD
# ENUMPC                                        PO827ER
# txt          HAYDEMHALLM11195791 CLARENCE ROADPO827ER
# Name: 17339, dtype: object

```

## Evaluate Results

Firstly, we need to prepare `true_blocks` DataFrame from our data (using known `person_id` in both datasets):

```python
# Create x and y indices
census['x'] = range(len(census))
cis['y'] = range(len(cis))

# Find true matches using person_id
true_blocks = pd.merge(
    left=census[['PERSON_ID', 'x']],
    right=cis[['PERSON_ID', 'y']],
    on='PERSON_ID'
)

# Add block numbers
true_blocks['block'] = range(len(true_blocks))

true_blocks.shape
# (24043, 4)
```
Let's sample 1000 pairs for which we will evaluate:
```python
matches = true_blocks.sample(1000, random_state=42)
```

Now we can evaluate the algorithm:

```python
eval_result = blocker.eval(rec_lin_result, matches[['x', 'y', 'block']])
```

and print the evaluation metrics:

```python
print(eval_result.metrics)
# recall         0.997000
# precision      1.000000
# fpr            0.000000
# fnr            0.003000
# accuracy       0.999997
# specificity    1.000000
# f1_score       0.998498
```

**NOTE:** Keep in mind that the metrics shown above are based only on the records that appear in `true_blocks`.
We assume that we have no knowledge
about the other records and their true blocks.


For this example, using `hnsw` we achieve:

- `99.7%` recall and `100%` precision
- close to `100%` accuracy
- Great reduction ratio of `0.999961`
- Most blocks contain just 2-3 records

This demonstrates BlockingPy's effectiveness at finding matching records while drastically reducing the number of required comparisons.