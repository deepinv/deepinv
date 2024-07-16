# Outline for the pseudorandom paper

## Title
- Pseudorandom Phase Retrieval: [why it's useful]

## Abstract
- context (random phase retrieval)
- problem (slow)
- solution (pseudorandom pr)
- results:
    - cosine similarity
    - number of layers
    - forward time

## Introduction
- phase retrieval:
    - history
    - applications
    - types of the sensing matrix
- random phase retrieval:
    - what it is
    - application
    - simplicity for analysis
    - problem: too slow
- contribution: pseudorandom pr
- structure:
    - backgrounds: random pr and reconstruction methods
    - pseudorandom pr
    - results
    - conclusion

## Background
1. random pr:
    - formulation
    - results we have
    - algorithms to solve
1. spectral method:
    - formulation
    - power iteration
1. gradient descent:
    - grandient formula
    - other algorithms: AMP, Projection, convex relaxation

## Pseudorandom Phase Retrieval
- formulation
- intuition
- how to downsample and upsample
- 2d fft

## Results
1. technical details:
    - algorithms
    - random and pseudonrandom
1. cosine similarity:
    - large variance and low similarity for gd random
    - steeper curve for pseudorandom
    - variance quickly decreases after oversampling 2
1. effects of number of layers:
    - big leap from 1 layer to 2, then saturate
1. forward time:
    - constant time on gpu for pseudorandom

## Conclusion:
- We proposed pseudorandom pr
- What it can do
- code availability
- future work:
    - try different sampling distributions (Haar)
    - investigate change from 1 layer to 2
    - hardware experiments
