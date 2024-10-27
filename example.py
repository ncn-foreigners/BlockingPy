# Example usage:
from blockingpy import blocker
import pandas as pd

x = pd.DataFrame({
    'txt': [
        "jankowalski21/10/1991",
        "kowalskijan/12/01/1919",
        "kowalskimjan/17/01/1991",
        "kowaljan15/01/1991",
        "montypython/15/01/1991",
        "pythonmonty/15/01/1991",
        "cyrkmontypython/15/01/1019",
        "monty/15/01/1991"
    ]
})

y = pd.DataFrame({
    'txt': [
        "montypython/15/02/1991",
        "kowalskijan/15/03/1919",
        "other/10000/12"
    ]
})

control_txt = {
    'n_shingles' : 3,
    'lowercase' : True,
    'strip_non_alphanum' : True,
    'n_chunks' : 5000

  }
control_ann = {
    'annoy':{
        'distance': 'euclidean',
        'k_search': 7
    }
}
verbose = 2

blocker = blocker.Blocker()
result = blocker.block(x=x['txt'], y=y['txt'], control_txt=control_txt, verbose=2, control_ann=control_ann, ann='annoy')
print(result.result)