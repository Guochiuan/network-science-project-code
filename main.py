import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import numpy as np


H=nx.read_gml("lesmis_data.gml")

print(len(H.edges()))


print(nx.is_connected(H))
# H.edges
nodes = list(H.nodes())
pairs = []
s = dict(nx.shortest_path_length(H))
# s = nx.shortest_path_length(H)
for src in s:
    for des in s[src]:
        if src != des:
            pairs.append(s[src][des])
#             print(s[src][des])
plt.hist(pairs, bins=[0,1,2,3,4,5,6])
# plt.show()
m = max(pairs)

summation = 0
for p in pairs:
    summation+=p
print ("maximum: ", m)
average = summation / len(pairs)
print ("average: ", average)


value = nx.get_edge_attributes(H, "value")
# print(value[('Myriel', 'MlleBaptistine')])

# for src in nodes:
#     for des in nodes:
#         if (src, des) in value:
#             print(value[(src, des)])
#         else:
#             print(0)

# print(nodes)
# print("test")
# print(value[('Myriel', 'Napoleon')])
# print(value[('Napoleon', 'Myriel')])
matrix = []
for src in nodes:
    row = []
    s = 0
    for des in nodes:
        if (src, des) in value:
            s += value[(src, des)]
        elif (des, src) in value:
            s += value[(des, src)]
    for des in nodes:
        if (src, des) in value:
            row.append(value[(src, des)]/s)
        elif (des, src) in value:
            row.append(value[(des, src)]/s)
        else:
            row.append(0)
    matrix.append(row)

print("matrix")
# print(matrix[1])
print(matrix)
transition_matrix = np.array(matrix)

print("herereeerere")
print(matrix[0])
#
# kk = nx.adjacency_matrix(H, weight ='value')
# print(kk)



# print(transition_matrix[60])

# transition_matrix = np.array([[0,0.7,0.3, 0],
#                             [0.3,0,0.4, 0.3],
#                             [0.3,0.3,0.2, 0.2],
#                               [0,0.5,0.2,0.3]])

# A = [[6, 7],
#      [8, 9]]
#
# B = [[1, 3],
#      [5, 7]]
#
# print(np.dot(A, B))
#
# print("----------")
#
# print(np.dot(B, A))
'''
Since the sum of each row is 1, our matrix is row stochastic.
We'll transpose the matrix to calculate eigenvectors of the stochastic rows.
'''
# print(transition_matrix)
transition_matrix_transp = transition_matrix.T


a = np.array([1/len(nodes)]*len(nodes))
re = np.dot(transition_matrix_transp, a)
for i in range(100):
    re = np.dot(transition_matrix_transp, re)
print("this")
print(re)

for i, t in enumerate(zip(nodes, re)):
    print(i, t)

print(sorted(zip(re, nodes), reverse=True)[:3])










eigenvals, eigenvects = np.linalg.eig(transition_matrix_transp)
print("eigenvalue",eigenvals)
print(eigenvects)
'''
Find the indexes of the eigenvalues that are close to one.
Use them to select the target eigen vectors. Flatten the result.
'''
print("eigenvals", eigenvals)
close_to_1_idx = np.isclose(eigenvals,1)
print("close to 1", close_to_1_idx)
target_eigenvect = eigenvects[:,close_to_1_idx]
print("target_eigenvect", target_eigenvect)
target_eigenvect = target_eigenvect[:,0]
print("taget_e", target_eigenvect)
# Turn the eigenvector elements into probabilites
total = 0
for i in target_eigenvect:
    total += i
stationary_distrib = target_eigenvect / total

print(stationary_distrib)

for i, t in enumerate(zip(nodes, stationary_distrib)):
    print(i, t)

print(sorted(zip(stationary_distrib, nodes), reverse=True)[:3])