# Proving and Simulating the Johnson-Lindenstrauss Lemma
Project completed by Henry Smith, Max Ranis, and Andrew Wei

## Code Example:
Code and visualizations by Henry Smith

```
# import the necessary function
from jl_project import compute_jl_projection

# as well as numpy
import numpy as np
```

The `compute_jl_projection` function expects two inputs: 

1. a numpy ndarray whose rows are vectors in \mathbb{R}^d
2. a value of \epsilon \in (0,1) which controls the algorithm's tolerance for distance distortion in the lower-dimensional space \mathbb{R}^k

Note that smaller \epsilon implies greater preservation of the original \ell^2 distances between points in \mathbb{R}^d.

`compute_jl_projection` returns a tuple containing:

1. The dimension k of the subspace onto which we are projecting the vectors in \mathbb{R}^d
2. The vectors in \mathbb{R}^k which preserve the \ell^2 distances between the original vectors up to a factor of \epsilon

```
# Generate a dataset of 20 random points in \mathbb{R}^1000
n = 20
d = 1000
orig_pts = np.random.normal(0, 1, size=(n,d))
epsilon = 0.1 

# Run the algorithm
k, projected_pts = compute_jl_projection(pts, epsilon)
```

```
print(k)
>> 36
```
