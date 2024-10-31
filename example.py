from blockingpy import blocker
import pandas as pd
import time

x = pd.DataFrame({
    'txt': [
        "jankowalski",
        "kowalskijan",
        "kowalskimjan",
        "kowaljan",
        "montypython",
        "pythonmonty",
        "cyrkmontypython",
        "monty"
    ]
})

y = pd.DataFrame({
    'txt': [
        "montypython", 
        "kowalskijan", 
        "other",
    ]
})

# control_txt = {
#     'n_shingles' : 3,
#     'lowercase' : True,
#     'strip_non_alphanum' : True,
#     'n_chunks' : 5000

#   }
# control_ann = {
#     'nnd':{
#         'metric': 'euclidean',
#         'n_threads': None,
#         'tree_init': True,
#         'n_trees': None,
#         'leaf_size': None,
#         'pruning_degree_multiplier': 1.5,
        
#     },
# }
# verbose = 3

blocking = blocker.Blocker()
#blocking_result = blocker.block(x=x['txt'], y=y['txt'], deduplication=True, control_txt=control_txt, verbose=verbose, control_ann=control_ann, ann='nnd')
blocking_result = blocking.block(x=x['txt'])
print(blocking_result)
print(blocking_result.result)

# ========================================================
# Blocking based on the kd method.
# Number of blocks: 2
# Number of columns used for blocking: 28
# Reduction ratio: 0.5714
# ========================================================
# Distribution of the size of the blocks:
# Size -> 3 : 2 <- Number of blocks with this size
#    x  y  block      dist
# 0  0  1      0  1.414214
# 1  1  2      0  1.732051
# 2  1  3      0  2.236068
# 3  4  5      1  1.414214
# 4  4  6      1  2.000000
# 5  5  7      1  2.449490


# start_time = time.time()
# end_time = time.time()
# print(f"Execution time: {end_time-start_time:.2f} seconds.")