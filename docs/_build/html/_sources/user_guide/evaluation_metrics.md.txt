(evaluation_metrics)=
# Evaluation Metrics

In this section we explain the evaluation metrics used to assess blocking quality in BlockingPy.

## Notation and Terminology

In the context of blocking evaluation, we use the following notation:

### Basic Counts

- **TP** (True Positives): Number of record pairs correctly identified as matches - pairs that are both predicted matches and true matches, also known as Correct Links

- **TN** (True Negatives): Number of record pairs correctly identified as non-matches - pairs that are both predicted non-matches and true non-matches, also known as Correct Non-Links 

- **FP** (False Positives): Number of record pairs incorrectly identified as matches - pairs that are predicted matches but are true non-matches, also known as False Links

- **FN** (False Negatives): Number of record pairs incorrectly identified as non-matches - pairs that are predicted non-matches but are true matches, also known as False Non-Links



### Block-Related Notation
For deduplication:

- **n**: Total number of records in the dataset
- **$B_i$**: The i-th block
- **|$B_i$|**: Size (number of records) of block i
- **$\binom{n}{2}$**: Total number of possible record pairs in a dataset of size n

For record linkage:

- $\sum_{i} |B_{i,x}| \cdot |B_{i,y}|$ is the number of comparisons after blocking
- $|B_{i,x}|$ is the number of unique records from dataset X in i-th block
- $|B_{i,y}|$ is the number of unique records from dataset Y in i-th block
- $m$ and $n$ are the sizes of the two original datasets being linked

The blocking outcome can be represented in a confusion matrix as follows:

|               | True Match   | True Non-Match |
|---------------|------------------|---------------------|
| Predicted Match    | TP              | FP                 |
| Predicted Non-Match| FN              | TN                 |

## Evaluation Metrics

### Classification Metrics

#### Precision
Fraction of correctly identified pairs among all pairs predicted to be in the same block:

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

#### Recall
Fraction of actual matching pairs that were correctly identified:

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

#### F1 Score
Harmonic mean of precision and recall:

$$
\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

#### Accuracy
Fraction of all correct predictions:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

#### Specificity
Fraction of actual non-matching pairs correctly identified:

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

#### False Positive Rate (FPR)
Fraction of actual non-matching pairs incorrectly predicted as matches:

$$
\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}
$$

#### False Negative Rate (FNR)
Fraction of actual matching pairs incorrectly predicted as non-matches:

$$
\text{FNR} = \frac{FN}{FN + TP} = 1 - \text{Recall}
$$

### Blocking Efficiency Metrics

#### Reduction Ratio (RR)
Measures how effectively the blocking method reduces the number of comparisons needed. The formula differs for deduplication and record linkage scenarios:

For deduplication (comparing records within one dataset):

$
\text{RR}_{\text{dedup}} = 1 - \frac{\sum_{i} \binom{|B_i|}{2}}{\binom{n}{2}}
$

where:
- $\sum_{i} \binom{|B_i|}{2}$ is the number of comparisons after blocking
- $\binom{n}{2}$ is the total possible comparisons without blocking
- $n$ is the total number of records in the dataset

For record linkage (comparing records between two datasets):

$
\text{RR}_{\text{link}} = 1 - \frac{\sum_{i} |B_{i,x}| \cdot |B_{i,y}|}{m \cdot n}
$

where:
- $\sum_{i} |B_{i,x}| \cdot |B_{i,y}|$ is the number of comparisons after blocking
- $|B_{i,x}|$ is the number of unique records from dataset X in i-th block
- $|B_{i,y}|$ is the number of unique records from dataset Y in i-th block
- $m$ and $n$ are the sizes of the two original datasets being linked

A reduction ratio closer to 1 indicates greater reduction in the comparison space, while a value closer to 0 indicates less reduction.

## Important Considerations

When evaluating blocking performance, it's crucial to understand that not all metrics carry equal importance due to the nature of the blocking procedure. Blocking serves as a preliminary step in the record linkage/deduplication pipeline, designed to reduce the computational burden while maintaining the ability to find true matches in subsequent steps.

Key priorities in blocking evaluation should focus on:

- **Recall** : High recall is critical as any true matches missed during blocking cannot be recovered in later stages of the linkage process. A blocking method should prioritize maintaining high recall even if it means lower precision.
- **Reduction Ratio** : This metric is essential as it directly measures how effectively the blocking method reduces the computational complexity of the subsequent matching process.
- **FNR** : Critical as False Negative pairs can not be adressed in the later stages of entity matching procedure.

As for other metrics:

- **Accuracy and Specificity** : Those should usually be high since most pairs fall into the **FN** category due to the nature of blocking.

- **Precision** : Low precision scores would be adressed in the later stages of entity matching procedure as most False Positive pairs would be eliminated during one-to-one comparison.

- **F1 score and FPR** : Same reasons as above.

Therefore, when evaluating blocking results, focus on achieving high recall and a good reduction ratio while accepting that other metrics may show values that would be considered poor in a final matching context.