(blocklib_comp)=
# BlockingPy vs blocklib - comparison

Below we compare BlockingPy with blocklib, a similar library for blocking. We present results obtained by running algorithms from both libraries on 3 generated datasets. The datasets were generated using the `geco3` tool, which allows for controlled generation of datasets with duplicates. The datasets  resemble real-world personal information data with the fields such as name, 2nd name, surname, 2nd surname, dob, municipality, and country of origin. There are 1k, 10k and 100k records respectively, with 500, 5k and 50k duplicates in each dataset. For each original record, there are 0, 1, or 2 duplicates. The datasets and code to reproduce the results can be found [here](https://github.com/ncn-foreigners/BlockingPy/tree/main/benchmark). The results were obtained on 6 cores Intel i5 CPU with 16GB RAM (py 3.12).


| algorithm                   | dataset\_size | time\_sec |   recall | reduction\_ratio | pairs (M) |
| --------------------------- | ------------: | --------: | -------: | ---------------: | --------: |
| P-Sig                       |         1 500 |     0.067 | 0.599384 |         0.996124 |  0.004358 |
| λ-fold LSH                  |         1 500 |     0.191 | 0.426810 |         0.993112 |  0.007744 |
| BlockingPy (voyager)        |         1 500 |     0.276 | 0.947612 |         0.997341 |  0.002989 |
| BlockingPy (faiss\_hnsw)    |         1 500 |     0.342 | 0.959938 |         0.997533 |  0.002773 |
| BlockingPy (faiss\_lsh)     |         1 500 |     0.186 | 0.961479 |         0.997379 |  0.002947 |
| P-Sig                       |        15 000 |     0.461 | 0.616241 |         0.996380 |  0.407185 |
| λ-fold LSH                  |        15 000 |     1.949 | 0.420727 |         0.994069 |  0.667196 |
| BlockingPy (voyager)        |        15 000 |     6.540 | 0.883681 |         0.999646 |  0.039785 |
| BlockingPy (faiss\_hnsw)    |        15 000 |    11.266 | 0.913070 |         0.999726 |  0.030865 |
| BlockingPy (faiss\_lsh)     |        15 000 |     1.423 | 0.901160 |         0.999701 |  0.033592 |
| P-Sig                       |       150 000 |     3.336 | 0.608723 |         0.996424 | 40.231251 |
| λ-fold LSH                  |       150 000 |    19.415 | 0.412729 |         0.994050 | 66.933870 |
| BlockingPy (voyager)        |       150 000 |   107.014 | 0.732607 |         0.999944 |  0.632477 |
| BlockingPy (faiss\_hnsw)    |       150 000 |   245.565 | 0.832314 |         0.999967 |  0.376853 |
| BlockingPy (faiss\_lsh)     |       150 000 |    55.911 | 0.818214 |         0.999964 |  0.404749 |


## Why `BlockingPy` outperforms blocklib

1. **Much higher recall**

Across all datasets, `BlockingPy` achieves higher recall then `blocklib` algorithms. (~0.43 for `blocklib` vs ~0.88 for `BlockingPy`).

2. **Better reduction ratio**

`BlockingPy` achieves better reduction ratio than `blocklib` algorithms, while maintaining higher recall. For instance on a dataset of size 150_000 records the difference in number of pairs between RR of 0.994 (λ-fold LSH) and RR of 0.99994 (voyager) is a difference of 67 milion pairs vs. 0.65 milion pairs requiring comparison.

3. **Minimal setup versus manual tuning**

Results shown for BlockingPy can be obtained with just a few lines of code, e.g., `blocklib`'s p-sig algorithm requires manual setup of blocking features, filters, bloom-filter parameters and signature specifications, which could require significant time and effort to tune.

4. **Scalability**

`BlockingPy` algorithms allow for `n_threads` selection and most algorithms allow for on-disk index building, where `blocklib` is missing both of these fetures.

## Where is `blocklib` better

1. **Privacy preserving blocking**

`blocklib` implements privacy preserving blocking algorithms, which are not available in `BlockingPy`.

2. **Time**

`blocklib` finishes the *blocking* phase sooner, but the extra minutes that **BlockingPy** spends are quickly repaid in the *matching* phase.  
In our benchmark (150k dataset) `blocklib` left **≈ 67 million** candidate pairs, whereas BlockingPy left **≈ 0.65 million**, that's a **~100 ×** reduction.  
Even though BlockingPy’s blocking step is **~5 ×** slower, the downstream classifier now has **100 ×** less work, so the end-to-end pipeline could still be faster, while achieving much higher recall (0.73 vs. 0.41).


Additionally, we can tune the `voyager` algorithm to achieve similar recall as blocklib's algorithms. On those settings the time difference is only ~1.6x, while still getting ~37x less candidate pairs (67 million vs. 1.8 million) compared to λ-fold LSH.

| algorithm                   | dataset\_size | time\_sec |   recall | reduction\_ratio | pairs (M) |
| --------------------------- | ------------: | --------: | -------: | ---------------: | --------: |
| BlockingPy (voyager) - fast |       150 000 |    33.007 | 0.482704 |         0.999841 |  1.791272 |


## Blockingpy with GPU acceleration
We also ran BlockingPy on GPU (`ann="gpu_faiss"`; index types: flat, ivf, cagra). On these datasets the GPU variants match the CPU back-ends on recall and reduction ratio, and the blocking step is faster. For presented dataset sizes, GPU gains are not clearly visible, however on larger datasets our GPU version might surpass `blocklib`'s algorithms in terms of speed, while still achieving much higher recall.


| algorithm                     | dataset\_size | time\_sec |   recall | reduction\_ratio | pairs (M) |
| ----------------------------- | ------------: | --------: | -------: | ---------------: | --------: |
| BlockingPy (gpu\_faiss flat)  |       150 000 |    25.851 | 0.839217 |         0.999967 |  0.367493 |
| BlockingPy (gpu\_faiss ivf)   |       150 000 |    65.396 | 0.799889 |         0.999958 |  0.477616 |
| BlockingPy (gpu\_faiss cagra) |       150 000 |    48.155 | 0.825767 |         0.999966 |  0.378721 |
