import numpy as np

def compute_jl_projection(V, epsilon):
  """Projects the set of points V in \mathbb{R}^d onto a lower subspace such that the distance between any two points in V 
  is preserved by a factor of \epsilon; note that the dimension of the subspace onto which we project is a function of both
  the number of points in V as well as \epsilon"""

  if not isinstance(V, np.ndarray):
    raise TypeError("V must be a numpy nd array.")
  
  if not 0 < epsilon and epsilon < 1:
    raise ValueError("epsilon must be between 0 and 1, exclusive")

  # Compute the dimension of the subspace onto which we will project
  d = V.shape[1]
  n = V.shape[0]
  k = int(np.ceil((epsilon**2/2 - epsilon**3/3)**(-1)*np.log(n)))

  # Print the dimension of the space onto which we are projecting
  if k > d:
    raise ValueError(f"Projecting onto a space of dimension {k} from a space of dimension {d}.")
  else:
    print(f"Projecting onto a space of dimension {k}.")

  # Compute the original squared \ell^2 distances between points in V
  orig_distances = compute_distances(V)

  mismatches = n
  proj = np.zeros((k, n))
  iterator = 0

  while mismatches:
    # Count the number of iterations we have completed
    iterator += 1

    # Generate our [random] projection matrix
    A = (1/np.sqrt(k))*np.random.normal(0, 1, (k, d))

    # Perform the projection 
    proj = A @ V.T

    # Compute the squared \ell^2 distances between the points in the k-dimensional subspace
    proj_distances = compute_distances(proj.T) 

    # Compute the number of points that *do not* satisfy the desired distance distortion property
    mismatches = n**2 - np.sum(np.logical_and(proj_distances >= (1 - epsilon)*orig_distances,  proj_distances <= (1 + epsilon)*orig_distances))

    print(mismatches)

  print(f"Projection found after {iterator} iterations.")
  return proj

def compute_distances(M):
  """Computes the squared Euclidean distance between each pair of points in M,
   where the rows of M are assumed to be the points; returns an upper triangular
   matrix whose ijth entry is the [squared Euclidean] distance between points i and j."""

  # Get the number of points in our dataset
  n_points = int(M.shape[0])

  # Initialize the matrix of distances
  dist = np.zeros((n_points, n_points))

  for i in range(n_points):
    for j in range(i + 1, n_points):
      
      # Compute the distance between points i and j
      dist[i,j] = np.linalg.norm(M[i,:] - M[j,:],ord=2)**2
    
  # Return the matrix of distances
  return dist
