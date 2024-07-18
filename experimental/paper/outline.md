# Outline for the pseudorandom paper

## Title
- Pseudorandom Phase Retrieval: [why it's useful]

## Abstract
- context (random phase retrieval)
- problem (slow, difficult to realize experimentally)
- solution (pseudorandom pr)
- results (in an intuitive way, no technical details):
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
- contribution (to be detailed): pseudorandom pr
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
1. random models in practice:
    - CDI
    - random probe ptychography

## Reconstruction algorithms
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
    - plot
    - large variance and low similarity for gd random
    - steeper curve for pseudorandom
    - variance quickly decreases after oversampling 2
1. effects of number of layers:
    - plot
    - big leap from 1 layer to 2, then saturate
1. forward time:
    - plot
    - constant time on gpu for pseudorandom

## Conclusion:
- We proposed pseudorandom pr
- What it can do
- code availability
- future directions:
    - try different sampling distributions (Haar)
    - investigate change from 1 layer to 2
    - hardware experiments