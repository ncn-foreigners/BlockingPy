(blocklib_comp)=
# BlockingPy vs blocklib - comparison

Below we compare BlockingPy with blocklib, a similar library for blocking. We present results obtained by running algorithms from both libraries on 3 generated datasets. The datasets were generated using the `geco3` tool, which allows for controlled generation of datasets with duplicates. The datasets  resemble real-world personal information data with the fields such as name, 2nd name, surname, 2nd surname, dob, municipality, and country of origin. There are 1k, 10k and 100k records respectively, with 500, 5k and 50k duplicates in each dataset. For each original record, there are 0, 1, or 2 duplicates.

| Dataset size | Algorithm                     | Recall | Reduction Ratio | Blocking time (s) |
|--------------|------------------------------|:------:|:---------------:|:-----------------:|
| 1 500        | P-Sig (blocklib)             | 0.459 | 0.996 | 0.36 |
| 1 500        | λ-fold LSH (blocklib)        | 0.427 | 0.993 | 0.40 |
| 1 500        | **BlockingPy (faiss_hnsw)**  | **0.960** | **0.998** | 2.16 |
| 1 500        | **BlockingPy (faiss_lsh)**   | **0.961** | **0.997** | **0.53** |
| 1 500        | **BlockingPy (voyager)**     | **0.954** | **0.997** | 1.20 |
| 15 000       | P-Sig (blocklib)             | 0.452 | 0.997 | 0.64 |
| 15 000       | λ-fold LSH (blocklib)        | 0.420 | 0.994 | 2.94 |
| 15 000       | **BlockingPy (faiss_hnsw)**  | **0.913** | **0.999** | 51.70 |
| 15 000       | **BlockingPy (faiss_lsh)**   | **0.901** | **0.999** | **11.34** |
| 15 000       | **BlockingPy (voyager)**     | **0.895** | **0.999** | 28.76 |
| 150 000      | P-Sig (blocklib)             | 0.450 | 0.997 | 6.93 |
| 150 000      | λ-fold LSH (blocklib)        | 0.413 | 0.994 | 32.03 |
| 150 000      | **BlockingPy (faiss_hnsw)**  | **0.836** | **0.99996** | 1 020.05 |
| 150 000      | **BlockingPy (faiss_lsh)**   | **0.818** | **0.99996** | **865.63** |
| 150 000      | **BlockingPy (voyager)**     | **0.762** | **0.99995** | 648.07 |

## Why `BlockingPy` outperforms blocklib

1. **Much higher recall**

Across all datasets, `BlockingPy` achieves higher recall then `blocklib` algorithms. (~0.45 for `blocklib` vs ~0.9 for `BlockingPy`).

2. **Better reduction ratio**

`BlockingPy` achieves better reduction ratio than `blocklib` algorithms, while maintaining higher recall. For instance on a dataset of size 150_000 records the difference in number of pairs between RR of 0.994 ( λ-fold LSH) and RR of 0.99995 (voyager) is a difference of 66.5 milion pairs vs. 560 thousand pairs requiring comparison.

3. **Minimal setup versus manual tuning**

Results shown for BlockingPy can be obtained with just a few lines of code, e.g., `blocklib`'s p-sig algorithm requires manual setup of blocking features, filters, bloom-filter parameters and signature specifications, which could require significant time and effort to tune.

4. **Scalability**

`BlockingPy` algorithms allow for `n_threads` selection and most algorithms allow for on-disk index building, where `blocklib` is missing both of these fetures.

## Where is `blocklib` better

1. **Privacy preserving blocking**

`blocklib` implements privacy preserving blocking algorithms, which are not available in `BlockingPy`.

2. **Time**

`blocklib` finishes the *blocking* phase sooner, but the extra minutes that **BlockingPy** spends are quickly repaid in the *matching* phase.  
In our benchmark (150k dataset) `blocklib` left **≈ 66 million** candidate pairs, whereas BlockingPy left **≈ 0.56 million**, that's a **~120 ×** reduction.  
Even though BlockingPy’s blocking step is **~20 ×** slower, the downstream classifier now has **120 ×** less work, so the end-to-end pipeline could still be faster, while achieving much higher recall (0.76 vs. 0.41).


Additionally, we can tune the `voyager` algorithm to achieve similar recall as blocklib's algorithms. On those settings the time difference is only 7x, while still getting 30x less candidate pairs (66 million vs. 2.7 million).

| Dataset size | Algorithm                     | Recall | Reduction Ratio | Blocking time (s) |
|--------------|------------------------------|:------:|:---------------:|:-----------------:|
| 150000        | **BlockingPy (voyager)**             | 0.401 | 0.9997 | 228 |