(core_concepts)=
# Core Concepts

## What is Blocking?

Blocking is a technique to reduce the number of comparisons needed by:

- Grouping similar records into "blocks"
- Only comparing records within the same block
- Excluding comparisons between records in different blocks

A good blocking strategy should:

- Drastically reduce the number of comparisons needed
- Maintain high recall (not miss true matches)
- Be computationally efficient

## The ANN Solution

BlockingPy uses Approximate Nearest Neighbor (ANN) algorithms to implement efficient blocking. The process works by:

- Converting records into vector representations
- Using ANN algorithms to efficiently find similar vectors
- Grouping similar vectors into blocks

This approach has several advantages:

- Scales well to large datasets
- Works with both structured and text data
- Provides tunable trade-offs between speed and accuracy

## Key Components
The main components of BlockingPy are:

**Blocker**: The main class that handles:

- Data preprocessing
- ANN algorithm selection and configuration
- Block generation


**BlockingResult**: Contains the blocking results, including:

- Block assignments
- Distance metrics
- Quality measures


**Controls**: Configuration systems for:

- Text processing parameters
- ANN algorithm parameters