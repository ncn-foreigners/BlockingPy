(gpu)=
# GPU SUPPORT (`blockingpy-gpu`)

GPU (CUDA) accelerated version is available via `blockingpy-gpu` package. The available GPU indexes are from FAISS GPU (Flat, IVF, IVFPQ, CAGRA).

## Requirements

- OS: Linux or Windows 11 with WSL2 (Ubuntu)  
- Python: 3.10  
- GPU: Nvidia with driver supporting CUDA ≥ 12.4  
- Tools: conda/mamba + pip

To use the package install FAISS-GPU via conda/mamba, then install `blockingpy-gpu` with pip.

```bash
# 1) Env
mamba create -n blockingpy-gpu python=3.10 -y
conda activate blockingpy-gpu
conda config --env --set channel_priority flexible

# 2) Install FAISS GPU (nightly cuVS build) - this version was tested
mamba install -y \
  -c pytorch/label/nightly -c rapidsai -c conda-forge \
  "faiss-gpu-cuvs=1.11.0" "libcuvs=25.4.*"

# 3) Install BlockingPy and the rest of deps with pip (or poetry, uv etc.)
pip install blockingpy-gpu
```

## What’s included vs CPU build

### GPU backends

- FAISS-GPU: flat, ivf, ivfpq, cagra.

### CPU backends also available in blockingpy-gpu

- FAISS (CPU), HNSW (hnswlib), Voyager, Annoy, NND (pynndescent).

**Not included**

- mlpack backends (k-d tree, LSH) are not shipped in blockingpy-gpu.

### Distances

L2, cosine, inner product

## Index Configuration

Works the same as in CPU Blockingpy. Here are the defaults:

```python
control_ann = {
    "gpu_faiss": {
            "index_type": "flat", #ivf, ivfpq, cagra
            "k_search": 30,
            "distance": "cosine",
            "path": None,

            "ivf_nlist": 100,
            "ivf_nprobe": 10,

            "ivfpq_nlist": 100,
            "ivfpq_m": 8,
            "ivfpq_nbits": 8,
            "ivfpq_nprobe": 10,
            "ivfpq_useFloat16": False,
            "ivfpq_usePrecomputed": False,
            "ivfpq_reserveVecs": 0,
            "ivfpq_use_cuvs": False,

            "cagra": {
                "graph_degree": 64,
                "intermediate_graph_degree": 128,
                "build_algo": "ivf_pq",
                "nn_descent_niter": 20,
                "itopk_size": 64,
                "max_queries": 0,
                "algo": "auto",
                "team_size": 0,
                "search_width": 1,
                "min_iterations": 0,
                "max_iterations": 0,
                "thread_block_size": 0,
                "hashmap_mode": "auto",
                "hashmap_min_bitlen": 0,
                "hashmap_max_fill_rate": 0.5,
                "num_random_samplings": 1,
                "seed": 0x128394,
            },
        },
}
```

From here everything works the same as in `blockingpy`. You can pass `ann='gpu_faiss'` to the `block` method and pass the controls dict with `'index_type'` and that's all. You can find an example of `blockingpy-gpu` workflow [here](https://blockingpy.readthedocs.io/en/latest/examples/gpu_example_dedup.html)

For more info about FAISS GPU and the indexes see [here.](https://github.com/facebookresearch/faiss/wiki/Running-on-GPUs)