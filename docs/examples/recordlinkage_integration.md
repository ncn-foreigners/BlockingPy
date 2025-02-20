# Integration with recordlinkage package

In this example we aim to show how users can utilize blocking results achieved with BlockingPy and use them with the [recordlinkage](https://github.com/J535D165/recordlinkage) package. The [recordlinkage](https://github.com/J535D165/recordlinkage) allows for both blocking and one-to-one record linkage and deduplication. However, it is possible to transfer blocking results from BlockingPy and incorporate them in the full entity resolution pipeline.

This example will show deduplication of febrl1 dataset which comes buillt-in with [recordlinkage](https://github.com/J535D165/recordlinkage).

We aim to follow the [Data deduplication](https://recordlinkage.readthedocs.io/en/latest/guides/data_deduplication.html#Introduction) example available on the recordlinkage documentation website and substitute the blocking procedure with our own.

## Setup

Firstly, we need to install `BlockingPy` and `recordlinkage`:

```bash
pip install blockingpy recordlinkage
```

Import necessary components:

```python
import recordlinkage
from recordlinkage.datasets import load_febrl1
from blockingpy import Blocker
import pandas as pd
from itertools import combinations 
```

## Data preparation

`febrl1` dataset contains 1000 records of which 500 are original and 500 are duplicates. It containts fictitious personal information e.g. name, surname, adress.

```python
df = load_febrl1()
print(df.head())

#               given_name	 surnam     street_number   address_1         address_2	suburb	    postcode	state	date_of_birth	soc_sec_id
# rec_id										
# rec-223-org	NaN	         waller	    6	            tullaroop street  willaroo	st james    4011        wa	    19081209	    6988048
# rec-122-org	lachlan	         berry	    69	            giblin street     killarney	bittern	    4814        qld	    19990219	    7364009

```

Prepare data in a suitable format for blockingpy. For this we need to fill missing values and concat fields to the `txt` column:

```python
df = df.fillna('')
df['txt'] = df['given_name'] + df['surname'] + \
            df['street_number'] + df['address_1'] + \
            df['address_2'] + df['suburb'] + \
            df['postcode'] + df['state'] + \
            df['date_of_birth'] + df['soc_sec_id']
```

## Blocking

Now we can obtain blocks from `BlockingPy`:

```python
blocker = Blocker()
blocking_result = blocker.block(x=df['txt'])

print(res)
# ========================================================
# Blocking based on the faiss method.
# Number of blocks: 500
# Number of columns used for blocking: 1023
# Reduction ratio: 0.998999
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
#          2 | 500  
print(res.result.head())
#        x    y  block      dist
# 0    474    0      0  0.048376
# 1    330    1      1  0.038961
# 2    351    2      2  0.086690
# 3    290    3      3  0.024617
# 4    333    4      4  0.105662
```

## Integration

To integrate our results, we need to obtain `pd.MultiIndex` with indexes of desired indicies in each pair. For that we need to create additional variables:

```python
df['x'] = range(len(df))
df['id'] = df.index
```
Now, we can merge the blocking result with main dataframe to extract information about indexes in each block:

```python
df_merged = res.result.merge(
    df[['x', 'id']],
    on='x',
    how='left'
)
df_merged = df_merged.rename({'id':'id_x'})
df_merged = df_merged.merge(
    df[['x', 'id']],
    left_on='y',
    right_on='x',
    how='left'
).rename({'id':'id_y', 'x_x':'x'}, axis=1).drop(labels=['x_y'], axis=1)
df_merged.head()


# 	x	y	block	dist	        id_x        	id_y
# 0	474	0	0	0.048376	rec-223-dup-0	rec-223-org
# 1	330	1	1	0.038961	rec-122-dup-0	rec-122-org
# 2	351	2	2	0.086690	rec-373-dup-0	rec-373-org
# 3	290	3	3	0.024617	rec-10-org	rec-10-dup-0
# 4	333	4	4	0.105662	rec-227-dup-0	rec-227-org
```

In the next step we create `pd.MultiIndex` with candidate pairs. In deduplication tasks each record in a given block must be paired with every other record in this block:

```python
df_merged = df_merged[['id_x', 'id_y', 'block']]

block_groups = (
    df_merged.melt(id_vars=['block'], value_name='record')
      .drop_duplicates()
      .groupby('block')['record']
      .agg(list)
)

all_pairs = []
for records in block_groups:
    all_pairs.extend(combinations(sorted(records), 2))

x_indices, y_indices = zip(*all_pairs)

df_end = pd.MultiIndex.from_arrays(
    [x_indices, y_indices],
    names=['rec_id_1', 'rec_id_2']
)

print(df_end)
# MultiIndex([('rec-223-dup-0', 'rec-223-org'),
#             ('rec-122-dup-0', 'rec-122-org'),
#             ('rec-373-dup-0', 'rec-373-org'),
#             ( 'rec-10-dup-0',  'rec-10-org'),
#             ('rec-227-dup-0', 'rec-227-org'),
#             (  'rec-6-dup-0',   'rec-6-org'),
#               ....
```

***NOTE 1*** : This step is required since we need to create pairs from all records in each block and not just from rows in `df_merged`. Although in this example each block contain only 2 records we show the generalized method which works for every block distribution.

***NOTE 2*** : This is the example for deduplication. Keep in mind that for record linkage this step needs to be modified.

Finally, we can put the `df_end` with candidate pairs into the `recordlinkage` pipeline and execute one-to-one comparison:

```python
dfA = load_febrl1() # load original dataset once again for clean data
compare_cl = recordlinkage.Compare()

compare_cl.exact("given_name", "given_name", label="given_name")
compare_cl.string(
    "surname", "surname", method="jarowinkler", threshold=0.85, label="surname"
)
compare_cl.exact("date_of_birth", "date_of_birth", label="date_of_birth")
compare_cl.exact("suburb", "suburb", label="suburb")
compare_cl.exact("state", "state", label="state")
compare_cl.string("address_1", "address_1", threshold=0.85, label="address_1")

features = compare_cl.compute(df_end, dfA) # (candidate pairs from blockingpy, original dataset)

matches = features[features.sum(axis=1) > 3]
print(len(matches))
# 458 
# vs. 317 when blocking traditionally on 'given_name'
```
Comparison rules were adopted from the [orignal example](https://recordlinkage.readthedocs.io/en/latest/guides/data_deduplication.html#Introduction). 