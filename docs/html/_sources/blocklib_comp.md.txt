(blocklib_comp)=
# BlockingPy vs blocklib - comparison

Below we compare BlockingPy with blocklib, a similar library for blocking. We present results obtained by running algorithms from both libraries on 3 generated datasets across 10 runs. The datasets were generated using the `geco3` tool, which allows for controlled generation of datasets with duplicates. The datasets  resemble real-world personal information data with the fields such as name, 2nd name, surname, dob, municipality, and country of origin. There are 1,5k, 15k and 150k records respectively, with 500, 5k and 50k duplicates in each dataset. For each original record, there are 0, 1, or 2 duplicates. The datasets and code to reproduce the results can be found [here](https://github.com/ncn-foreigners/BlockingPy/tree/main/benchmark). The results were obtained on 6 cores Intel i5 CPU with 16GB RAM (py 3.12).


| Algorithm               |   Size | Time [s]        | Recall              | Reduction Ratio      | Pairs (M)       |
| :---------------------- | -----: | :-------------- | :------------------ | :------------------ | :-------------- |
| BlockingPy (faiss_hnsw) |   1500 | 0.164 ± 0.049   | 0.959938 ± 0.000000 | 0.997533 ± 0.000000 | 0.003 ± 0.000   |
| BlockingPy (faiss_lsh)  |   1500 | 0.185 ± 0.015   | 0.955470 ± 0.008391 | 0.997371 ± 0.000112 | 0.003 ± 0.000   |
| BlockingPy (voyager)    |   1500 | 0.293 ± 0.029   | 0.951772 ± 0.006036 | 0.997382 ± 0.000025 | 0.003 ± 0.000   |
| P-Sig                   |   1500 | 0.051 ± 0.010   | 0.599384 ± 0.000000 | 0.996124 ± 0.000000 | 0.004 ± 0.000   |
| λ-fold LSH              |   1500 | 0.193 ± 0.009   | 0.465794 ± 0.033199 | 0.990865 ± 0.002847 | 0.010 ± 0.003   |
| BlockingPy (faiss_hnsw) |  15000 | 12.029 ± 1.681  | 0.913009 ± 0.000108 | 0.999726 ± 0.000000 | 0.031 ± 0.000   |
| BlockingPy (faiss_lsh)  |  15000 | 1.398 ± 0.148   | 0.899706 ± 0.003506 | 0.999708 ± 0.000007 | 0.033 ± 0.001   |
| BlockingPy (voyager)    |  15000 | 6.777 ± 0.903   | 0.875978 ± 0.004538 | 0.999635 ± 0.000008 | 0.041 ± 0.001   |
| P-Sig                   |  15000 | 0.388 ± 0.087   | 0.616241 ± 0.000000 | 0.996380 ± 0.000000 | 0.407 ± 0.000   |
| λ-fold LSH              |  15000 | 2.016 ± 0.251   | 0.453983 ± 0.027622 | 0.991644 ± 0.002993 | 0.940 ± 0.337   |
| BlockingPy (faiss_hnsw) | 150000 | 250.012 ± 4.232 | 0.832092 ± 0.000507 | 0.999967 ± 0.000000 | 0.375 ± 0.001   |
| BlockingPy (faiss_lsh)  | 150000 | 56.544 ± 1.315  | 0.818256 ± 0.001381 | 0.999964 ± 0.000000 | 0.400 ± 0.005   |
| BlockingPy (voyager)    | 150000 | 110.341 ± 3.815 | 0.715393 ± 0.005450 | 0.999940 ± 0.000002 | 0.675 ± 0.021   |
| P-Sig                   | 150000 | 3.444 ± 0.091   | 0.608723 ± 0.000000 | 0.996424 ± 0.000000 | 40.231 ± 0.000  |
| λ-fold LSH              | 150000 | 19.848 ± 0.476  | 0.450190 ± 0.026086 | 0.991550 ± 0.003045 | 95.064 ± 34.252 |




## Why `BlockingPy` outperforms blocklib

1. **Much higher recall**

Across all datasets, `BlockingPy` achieves higher recall then `blocklib` algorithms. (~0.52 for `blocklib` vs ~0.88 for `BlockingPy`).

2. **Better reduction ratio**

`BlockingPy` achieves better reduction ratio than `blocklib` algorithms, while maintaining higher recall. For instance on a dataset of size 150_000 records the difference in number of pairs between RR of 0.992 (λ-fold LSH) and RR of 0.99994 (voyager) is a difference of 95 milion pairs vs. 0.67 milion pairs requiring comparison.

3. **Minimal setup versus manual tuning**

Results shown for BlockingPy can be obtained with just a few lines of code, e.g., `blocklib`'s p-sig algorithm requires manual setup of blocking features, filters, bloom-filter parameters and signature specifications, which could require significant time and effort to tune.

4. **Scalability**

`BlockingPy` algorithms allow for `n_threads` selection and most algorithms allow for on-disk index building, where `blocklib` is missing both of these fetures.

## Where is `blocklib` better

1. **Privacy preserving blocking**

`blocklib` implements privacy preserving blocking algorithms, which are not available in `BlockingPy`.

2. **Time**

`blocklib` finishes the *blocking* phase sooner, but the extra minutes that **BlockingPy** spends are quickly repaid in the *matching* phase.  
In our benchmark (150k dataset) `blocklib` left **≈ 95 million** candidate pairs, whereas BlockingPy left **≈ 0.67 million**, that's a **~140 ×** reduction.  
Even though BlockingPy’s blocking step is **~5 ×** slower, the downstream classifier now has **140 ×** less work, so the end-to-end pipeline could still be faster, while achieving much higher recall (0.72 vs. 0.45).


Additionally, we can tune the `voyager` algorithm to achieve similar recall as blocklib's algorithms. On those settings the time difference is only ~1.8x, while still getting ~47x less candidate pairs (95 million vs. 2.0 million) compared to λ-fold LSH.

| Algorithm                   |   Size | Time [s]       | Recall              | Reduction Ratio     | Pairs (M)     |   |
| :-------------------------- | -----: | :------------- | :------------------ | :------------------ | :------------ | - |
| BlockingPy (voyager) - fast |   1500 | 0.215 ± 0.015  | 0.921726 ± 0.010369 | 0.996613 ± 0.000181 | 0.004 ± 0.000 |   |
| BlockingPy (voyager) - fast |  15000 | 2.495 ± 0.173  | 0.713395 ± 0.017146 | 0.999297 ± 0.000041 | 0.079 ± 0.005 |   |
| BlockingPy (voyager) - fast | 150000 | 34.395 ± 0.722 | 0.450416 ± 0.016316 | 0.999820 ± 0.000009 | 2.025 ± 0.102 |   |



## Blockingpy with GPU acceleration
We also ran BlockingPy on GPU (`ann="gpu_faiss"`; index types: flat, ivf, cagra). On these datasets the GPU variants match the CPU back-ends on recall and reduction ratio, and the blocking step is faster. For presented dataset sizes, GPU gains are not clearly visible, however on larger datasets our GPU version might surpass `blocklib`'s algorithms in terms of speed, while still achieving much higher recall.


| Algorithm                    |   Size | Time [s]       | Recall              | Reduction Ratio     | Pairs (M)     |
| :--------------------------- | -----: | :------------- | :------------------ | :------------------ | :------------ |
| BlockingPy (gpu_faiss cagra) |   1500 | 2.905 ± 0.327  | 0.959938 ± 0.000000 | 0.997499 ± 0.000000 | 0.003 ± 0.000 |
| BlockingPy (gpu_faiss flat)  |   1500 | 0.898 ± 0.528  | 0.963020 ± 0.000000 | 0.997514 ± 0.000000 | 0.003 ± 0.000 |
| BlockingPy (gpu_faiss ivf)   |   1500 | 0.908 ± 0.062  | 0.944992 ± 0.004238 | 0.997176 ± 0.000104 | 0.003 ± 0.000 |
| BlockingPy (gpu_faiss cagra) |  15000 | 6.835 ± 0.629  | 0.912838 ± 0.000294 | 0.999724 ± 0.000000 | 0.031 ± 0.000 |
| BlockingPy (gpu_faiss flat)  |  15000 | 1.662 ± 0.165  | 0.913380 ± 0.000000 | 0.999724 ± 0.000000 | 0.031 ± 0.000 |
| BlockingPy (gpu_faiss ivf)   |  15000 | 2.376 ± 0.244  | 0.866605 ± 0.003575 | 0.999627 ± 0.000006 | 0.042 ± 0.001 |
| BlockingPy (gpu_faiss cagra) | 150000 | 51.084 ± 0.802 | 0.826633 ± 0.000155 | 0.999966 ± 0.000000 | 0.380 ± 0.000 |
| BlockingPy (gpu_faiss flat)  | 150000 | 22.511 ± 0.508 | 0.839217 ± 0.000000 | 0.999967 ± 0.000000 | 0.367 ± 0.000 |
| BlockingPy (gpu_faiss ivf)   | 150000 | 71.458 ± 2.185 | 0.801427 ± 0.003905 | 0.999957 ± 0.000002 | 0.487 ± 0.021 |

