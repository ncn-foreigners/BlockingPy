(gpu_example_dedup)=
# Deduplication on GPU (adaptation of Example No. 2)

This example reproduces the [Deduplication No. 2](https://blockingpy.readthedocs.io/en/latest/examples/deduplication_2.html)
 walkthrough but with the GPU build of BlockingPy.
The GPU version (`blockingpy-gpu`) accelerates blocking with `FAISS-GPU`
, offering significant speedups on large datasets.  

## Installation

You cannot get FAISS-GPU from PyPI wheels directly, so installation requires conda/mamba for FAISS and pip for BlockingPy:

```bash
# 1) Create environment
mamba create -n blockingpy-gpu python=3.10 -y
conda activate blockingpy-gpu

# 2) Install FAISS GPU (nightly cuVS build) - this was tested
mamba install -c pytorch/label/nightly \
  faiss-gpu-cuvs=1.11.0=py3.10_ha3bacd1_55_cuda12.4.0_nightly -y

# 3) Install BlockingPy GPU package
pip install blockingpy-gpu
```

## Data preparation

Firstly, we need to prepare the dataset:

```python
import pandas as pd
from blockingpy import Blocker
from blockingpy.datasets import load_deduplication_data

data = load_deduplication_data()
data = data.fillna('')
data[['by', 'bm', 'bd']] = data[['by', 'bm', 'bd']].astype(str)

data['txt'] = (
    data["fname_c1"] +
    data["fname_c2"] +
    data['lname_c1'] +
    data['lname_c2'] +
    data['by'] +
    data['bm'] +
    data['bd']
)
```

## Deduplication

Now, we can deduplicate the dataset using `ann='gpu_faiss'`:

```python
blocker = Blocker()
dedup_result = blocker.block(
    x=data['txt'],
    ann='gpu_faiss',
    verbose=1,
    random_seed=42,
)
print(dedup_result)
# ========================================================
# Blocking based on the gpu_faiss method.
# Number of blocks: 2737
# Number of columns created for blocking: 674
# Reduction ratio: 0.999583
# ========================================================
# Distribution of the size of the blocks:
# Block Size | Number of Blocks
#          2 | 965            
#          3 | 722            
#          4 | 421            
#          5 | 247            
#          6 | 140            
#          7 | 98             
#          8 | 49             
#          9 | 35             
#         10 | 26             
#         11 | 13             
#         12 | 7              
#         13 | 3              
#         14 | 4              
#         15 | 1              
#         16 | 1              
#         17 | 1              
#         18 | 2              
#         20 | 1              
#         66 | 1  
print(dedup_result.result.head())
#       x  y  block      dist
# 0  3402  0      0  0.128420
# 1  1179  1      1  0.165676
# 2  2457  2      2  0.104868
# 3  1956  3      3  0.042670
# 4  4448  4      4  0.187500 
```

## Customizing GPU Index with control_ann

We can customize `gpu_faiss` through the `control_ann` dict. Let's set the algorithm to `cagra`:

```python
gpu_controls = {
    "gpu_faiss": {
        "index_type": "cagra",   # flat, ivf, ivfpq, cagra
        "distance": "cosine",
        # here you can tweak the parameters of CAGRA and others.
    }
}

blocker = Blocker()
dedup_result = blocker.block(
    x=data['txt'],
    ann='gpu_faiss',
    control_ann=gpu_controls,
    verbose=1,
    random_seed=42,
)
```

## Evaluation

Now, we can evaluate the algorithm. For that we need to prepare the `true_blocks` ground-truth dataset:

```python
df_eval = data.copy()
df_eval['block'] = df_eval['true_id']
df_eval['x'] = range(len(df_eval))

true_blocks_dedup = df_eval[['x', 'block']]
```
And now, evaluate it:

```python
blocker = Blocker()
eval_result = blocker.block(
    x=df_eval['txt'],
    ann='gpu_faiss',
    true_blocks=true_blocks_dedup,
    control_ann=gpu_controls,   # evaluation with chosen GPU index
    verbose=1,
    random_seed=42,
)

print(eval_result.reduction_ratio)
# 0.9995822182218221
print(eval_result.metrics)
# recall         1.000000
# precision      0.047895
# fpr            0.000398
# fnr            0.000000
# accuracy       0.999602
# specificity    0.999602
# f1_score       0.091412
# dtype: float64
print(eval_result.confusion)
#                  Predicted Positive  Predicted Negative
# Actual Positive                1000                   0
# Actual Negative               19879            49974121
```

When evaluated with the ground truth, `CAGRA` achieves very high recall (in this dataset `100%`), meaning no true duplicate pairs are lost, while still reaching an excellent reduction ratio (`99.95%`).
`CAGRA` is conceptually similar to `HNSW—both` are graph-based ANN algorithms—but unlike `HNSW` (CPU), `CAGRA` is fully GPU-optimized, allowing much higher throughput on large, high-dimensional datasets.


Note: this dataset is too small to demonstrate the speed advantage; the benefits of CAGRA become clear on larger inputs where GPU parallelism matters.

We encourage you to try `blockingpy-gpu` yourself!