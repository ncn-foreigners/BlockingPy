(blocklib_comp)=
# BlockingPy vs blocklib - comparison

Below we compare BlockingPy with blocklib, a similar library for blocking. We present results obtained by running algorithms from both libraries on 3 generated datasets. The datasets were generated using the `geco3` tool, which allows for controlled generation of datasets with duplicates. The datasets  resemble real-world personal information data with the fields such as name, 2nd name, surname, 2nd surname, dob, municipality, and country of origin. There are 1k, 10k and 100k records respectively, with 500, 5k and 50k duplicates in each dataset. For each original record, there are 0, 1, or 2 duplicates. The datasets and code to reproduce the results can be found [here](https://github.com/ncn-foreigners/BlockingPy/tree/main/benchmark). The results were obtained on 6 cores Intel i5 CPU with 16GB RAM.


| algorithm                   | dataset\_size | time\_sec |   recall | reduction\_ratio | pairs (M) |
| --------------------------- | ------------: | --------: | -------: | ---------------: | --------: |
| P-Sig                       |         1 500 |     0.067 | 0.459168 |         0.996371 |  0.004080 |
| λ-fold LSH                  |         1 500 |     0.210 | 0.426810 |         0.993112 |  0.007744 |
| BlockingPy (voyager)        |         1 500 |     0.545 | 0.949153 |         0.997395 |  0.002929 |
| BlockingPy (faiss\_hnsw)    |         1 500 |     0.435 | 0.959938 |         0.997517 |  0.002791 |
| BlockingPy (faiss\_lsh)     |         1 500 |     0.465 | 0.961479 |         0.997379 |  0.002947 |
| P-Sig                       |        15 000 |     0.507 | 0.451508 |         0.996838 |  0.355714 |
| λ-fold LSH                  |        15 000 |     2.241 | 0.420727 |         0.994069 |  0.667196 |
| BlockingPy (voyager)        |        15 000 |     8.363 | 0.881052 |         0.999647 |  0.039710 |
| BlockingPy (faiss\_hnsw)    |        15 000 |    13.714 | 0.913380 |         0.999725 |  0.030988 |
| BlockingPy (faiss\_lsh)     |        15 000 |     3.263 | 0.901160 |         0.999701 |  0.033592 |
| P-Sig                       |       150 000 |     4.657 | 0.449721 |         0.996871 | 35.202722 |
| λ-fold LSH                  |       150 000 |    20.703 | 0.412729 |         0.994050 | 66.933870 |
| BlockingPy (voyager)        |       150 000 |   211.529 | 0.721153 |         0.999942 |  0.656770 |
| BlockingPy (faiss\_hnsw)    |       150 000 |   343.390 | 0.832423 |         0.999966 |  0.377265 |
| BlockingPy (faiss\_lsh)     |       150 000 |   154.186 | 0.818230 |         0.999964 |  0.404709 |


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
Even though BlockingPy’s blocking step is **~10 ×** slower, the downstream classifier now has **100 ×** less work, so the end-to-end pipeline could still be faster, while achieving much higher recall (0.72 vs. 0.41).


Additionally, we can tune the `voyager` algorithm to achieve similar recall as blocklib's algorithms. On those settings the time difference is only 7x, while still getting ~37x less candidate pairs (67 million vs. 1.8 million).

| algorithm                   | dataset\_size | time\_sec |   recall | reduction\_ratio | pairs (M) |
| --------------------------- | ------------: | --------: | -------: | ---------------: | --------: |
| BlockingPy (voyager) – fast |       150 000 |   142.010 | 0.483153 |         0.999841 |  1.785544 |