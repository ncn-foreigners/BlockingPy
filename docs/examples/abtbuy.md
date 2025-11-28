(abt_buy)=
# Abt–Buy record linkage

This example shows how to use BlockingPy for record linkage on the **Abt–Buy**
product datasets. We:

- load Abt, Buy and ground truth files,
- build a simple text field,
- run embedding-based blocking with HNSW,

The datasets can be found [in the PyJedAI repository.](https://github.com/AI-team-UoA/pyJedAI/tree/main/data/ccer/D2)

## Setup

```bash
pip install blockingpy
```

## Load data

```python
from blockingpy import Blocker
import pandas as pd

abt = pd.read_csv("abt.csv", sep="|")
buy = pd.read_csv("buy.csv", sep='|')
gt = pd.read_csv("gt.csv", sep="|")

print(abt.head())

print(abt.shape)
print(buy.shape)
print(gt.shape)

#    id                                               name  \
# 0   0                          Sony Turntable - PSLX350H   
# 1   1  Bose Acoustimass 5 Series III Speaker System -...   
# 2   2                             Sony Switcher - SBV40S   
# 3   3                   Sony 5 Disc CD Player - CDPCE375   
# 4   4  Bose 27028 161 Bookshelf Pair Speakers In Whit...   

#                                          description  price  
# 0  Sony Turntable - PSLX350H/ Belt Drive System/ ...    NaN  
# 1  Bose Acoustimass 5 Series III Speaker System -...  399.0  
# 2  Sony Switcher - SBV40S/ Eliminates Disconnecti...   49.0  
# 3  Sony 5 Disc CD Player- CDPCE375/ 5 Disc Change...    NaN  
# 4  Bose 161 Bookshelf Speakers In White - 161WH/ ...  158.0


# (1076, 4)
# (1076, 4)
# (1076, 2)
```

## Creating "True Blocks"
We need to adjust the `gt` dataframe to match the expected format.

```python
gt['block'] = range(len(gt))
gt = gt.rename(columns={"D1": 'x', "D2": 'y'})
```

## Data preprocessing
We will convert all string columns to the `string` dtype and fill missing values. Then, we will create a new text field `name_price` by concatenating the `name` and `price` columns.

You can experiment with different combinations of fields to see how they affect blocking performance.
```python
str_cols = [col for col in abt.columns if col != 'id']
abt = abt.astype({col: 'string' for col in str_cols})
buy = buy.astype({col: 'string' for col in str_cols})

abt = abt.fillna('')
buy = buy.fillna('')

abt['name_price'] = abt['name'] + abt['price']
buy['name_price'] = buy['name'] + buy['price']
```

## Blocking with HNSW
We will use embedding-based blocking with HNSW.
```python
blocker = Blocker()

control_txt = {
        "encoder": "embedding",
        "embedding": {
            "model": "minishlab/potion-base-32M",
            "normalize": True,
            "max_length": 512,
            "emb_batch_size": 1024,
            "show_progress_bar": True,
            "use_multiprocessing": True,
            "multiprocessing_threshold": 10000,
        },
}

res = blocker.block(
    x=abt['name_price'],
    y=buy['name_price'],
    true_blocks=gt,
    verbose=1,
    random_seed=42,
    ann='hnsw',
    control_txt=control_txt,

)

print(res)
# INFO - ===== creating tokens =====
# 100%|██████████| 2/2 [00:00<00:00, 69.95it/s]
# 100%|██████████| 2/2 [00:00<00:00, 64.36it/s]
# INFO - ===== starting search (hnsw, x, y: 1076,1076, t: 512) =====
# INFO - ===== creating graph =====
# INFO - ===== evaluating =====
# ========================================================
# Blocking based on the hnsw method.
# Number of blocks: 902
# Number of columns created for blocking: 512
# Reduction ratio: 0.999071
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
#          2 | 787            
#          3 | 85             
#          4 | 16             
#          5 | 8              
#          6 | 3              
#          8 | 1              
#          9 | 1              
#         10 | 1              
# ========================================================
# Evaluation metrics (standard):
# recall : 82.342
# precision : 82.342
# fpr : 0.0164
# fnr : 17.658
# accuracy : 99.9672
# specificity : 99.9836
# f1_score : 82.342
print(res.confusion)
#                  Predicted Positive  Predicted Negative
# Actual Positive                 886                 190
# Actual Negative                 190             1156510
```
