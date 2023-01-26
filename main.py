import numpy as np
transition_matrix = np.array([[0,0.7,0.3, 0],
                            [0.3,0,0.4, 0.3],
                            [0.3,0.3,0.2, 0.2],
                              [0,0.5,0.2,0.3]])
'''
Since the sum of each row is 1, our matrix is row stochastic.
We'll transpose the matrix to calculate eigenvectors of the stochastic rows.
'''
transition_matrix_transp = transition_matrix.T
eigenvals, eigenvects = np.linalg.eig(transition_matrix_transp)
'''
Find the indexes of the eigenvalues that are close to one.
Use them to select the target eigen vectors. Flatten the result.
'''
close_to_1_idx = np.isclose(eigenvals,1)
target_eigenvect = eigenvects[:,close_to_1_idx]
target_eigenvect = target_eigenvect[:,0]
# Turn the eigenvector elements into probabilites
stationary_distrib = target_eigenvect / sum(target_eigenvect)

print(stationary_distrib)