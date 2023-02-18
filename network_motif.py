from itertools import permutations

import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import numpy as np
import pandas as pd
from networkx.algorithms import bipartite
from numpy import random, ceil, Inf
from numpy.testing import assert_equal
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from powerlaw import *



G = nx.DiGraph()

G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(3,1)

print("sdfsdfsdf", nx.triad_type(G))
# print(G)
# for (e1, e2, e3) in permutations(G.edges(), 3):
#     # print(set(e1))
#     if set(e1) == set(e2):
#         if e3[0] in e1:
#             print("111U")
#                 # e3[1] in e1:
#             print("111D")
#     elif set(e1).symmetric_difference(set(e2)) == set(e3):
#         if {e1[0], e2[0], e3[0]} == set(G.nodes()):
#             print("030C")
#                 # e3 == (e1[0], e2[1]) and e2 == (e1[1], e3[1]):
#         else:
#             print("030T")


for (e1, e2, e3) in permutations(G.edges(), 3):
    if set(G.nodes) == {e1[0], e2[0], e3[0]}:
        print("030c")
        break
    # if e3 == (e1[0], e2[1]) and e2 == (e1[1], e3[1]):
    if e1 == (e3[0], e2[0]) and e2 == (e1[1], e3[1]):
        print(e1)
        print(e2)
        print(e3)
        print("030t")
        break

    # # print(set(e1))
    # if set(e1) == set(e2):
    #     if e3[0] in e1:
    #         print("111U")
    #             # e3[1] in e1:
    #         print("111D")
    # if set(e1).symmetric_difference(set(e2)) == set(e3):
    #     if {e1[0], e2[0], e3[0]} == set(G.nodes()):
    #         print("030C")
    #             # e3 == (e1[0], e2[1]) and e2 == (e1[1], e3[1]):
    #     else:
    #         print("030T")

# G = nx.read_edgelist("blog.txt",create_using=nx.MultiGraph(), nodetype = int)
#
# a = [1,2,3,4,3,3]
# b = [2,3,5,5,6,7]
#
#
# # nx.degree_assortativity_coefficient(G)
# degree_list = list(dict(G.degree).values())
#
# # print("listt leng = ", len(degree_list))
# # degree_list = dict(G.degree)
# # print(degree_list)
# average_neighbor_degree_list=list(nx.average_neighbor_degree(G).values())
#
#
# degree_list = a
# average_neighbor_degree_list = b
#
#
# total = dict()
# for n in range(len(degree_list)):
#     if degree_list[n] not in total:
#         temp = list()
#         temp.append(average_neighbor_degree_list[n])
#         total[degree_list[n]] = temp
#     else:
#         total[degree_list[n]].append(average_neighbor_degree_list[n])
#
#
# print("sdfsdfdsfdfs = ",len(total))
# print(total)
#
# for key in total:
#     total[key] = sum(total[key])/len(total[key])
#
#
# print(total)
#
#
# # print("lennn", len(average_neighbor_degree_list))
# # print(average_neighbor_degree_list)
# plt.scatter(list(total.keys()), list(total.values()), marker='.')
# # plt.scatter(degree_list, average_neighbor_degree_list)
# plt.title('Friendship Paradox')
# plt.xlabel('Degree k')
# plt.ylabel('Average neighbor degree ')
# plt.show()
# result = sp.stats.pearsonr(degree_list, average_neighbor_degree_list)
# r = result[0]
# print("Pearson coreelatikon coefficient =", r)
# n = len(G.nodes)
# t = r *pow(n -2, .5)/ pow( (1-r*r),.5)
# print("t-value = ", t)
# degrees_of_freedom = len(G.nodes) - 2
# pval = sp.stats.t.sf(np.abs(t), df=degrees_of_freedom)*2
# print("p-value = ", pval)
# print("p-value is close to ", f"{pval:3.4f}")
#
#

# G = nx.gnp_random_graph(100, 0.02, seed=10374196)
# print(G.degree()[0])
# degree_sequence = sorted((d for n, d in G.degree()), reverse=True)

import numpy as np
from matplotlib import pyplot as plt
#
# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True
#
# y = np.random.standard_normal((20, 2))
# # y[:, 0] = y[:, 0] * 100
# # x = y.cumsum(axis=1)
# print(y)
# # print(x)
# plt.figure(figsize=(10, 6))
# plt.title('A Simple Plot')
# # fig, ax1 = plt.subplots()
#
#
# fig, ax = plt.subplots(figsize=(10, 6))
# plt.boxplot(y)
# plt.setp(ax, xticklabels=['1st', '2nd'])
#
# # plt.setp(ax, xticklabels=['1st', '2nd'])
# plt.xlabel('data set')
#
#
#
# # plt.subplot(211)
# # plt.plot(y[:, 0])
# # plt.title('A Simple Plot1')
# # # ax2 = ax1.twinx()
# #
# # plt.subplot(212)
# # plt.plot(y[:, 1], lw=1.5)
# # # plt.plot(y, 'ro')
# # plt.xlabel('index')
# # plt.ylabel('value')
# # plt.title('A Simple Plot2')
# # plt.ax.remove()
# plt.show()

# G = nx.read_edgelist("blog.txt",create_using=nx.DiGraph(), nodetype = int)
# out_degree_list = list(dict(G.out_degree).values())
# out_degree_list_excluding_zero = [i for i in out_degree_list if i != 0]
#
#
# # powerlaw.plot_pdf(out_degree_list_excluding_zero, linestyle='None', marker='o', color='b')
#
# ax = powerlaw.plot_pdf(out_degree_list_excluding_zero, linear_bins=True, linestyle='None', marker='o', color='b')
# ax.set_xscale("linear")
# ax.set_yscale("linear")
# plt.show()

# from scipy import *
# from matplotlib.pyplot import *
# matplotlib.rcParams.update({'font.size': 20})
#
# Nnodes=10000
# power=-2;
# maxdegree=1000;
# mindegree=1;
# ks = ((maxdegree**(power+1)-mindegree**(power+1) )*random.random(Nnodes)+mindegree**(power+1))**(1/(power + 1))
#
# [counts,bins,patches]=hist(ks,bins=100)
#
# figure()
# subplot(2,1,1)
# bar(bins[:-1],counts/float(sum(counts)),width=bins[1]-bins[0])
# ylabel("fraction of nodes")
#
# subplot(2,1,2)
# bar(bins[:-1],counts/float(sum(counts)),width=bins[1]-bins[0],log=True)
# #hist(ks,bins=arange(min(ks),max(ks)),normed=True,log=True)
# xlabel("degree")
# ylabel("fraction of nodes")
# plt.show()
# savefig("power_law_degree_distribution.png", transparent=True, dpi=60)
#
# maxdegfound=int(ceil(max(bins)))
# [counts,bins,patches]=hist(ks,bins=maxdegfound)
#
#
# countsnozero=counts*1.
# countsnozero[counts==0]=-Inf
#
# figure()
# scatter(bins[:-1],countsnozero/float(sum(counts)),s=60)
# yscale('log')
# xscale('log')
# ylim(0.00008,1.1)
# xlim(0.8,1100)
# xlabel('degree')
# ylabel("fraction of nodes")
# subplots_adjust(bottom=0.15)
# plt.show()
# savefig("power_law_degree_distribution_scatter.png", transparent=True, dpi=60)


# print("sdfsdfsdfdsfdfs")
# print(degree_sequence)
# dmax = max(degree_sequence)
# print("dmax")
# print(dmax)
#
# e = [('a','b'),('a','d'),('b','d'),('c','d'),('b','c'),('d','e'),('c','e')]
# G = nx.Graph(e)
# nx.draw(G, pos=nx.spring_layout(G, k =1), with_labels=True)
# nx.draw(G, pos=nx.spring_layout(G, k =1))
#
# result = nx.clustering(G)
# print (result.values())
# sum = 0
# for el in result.values():
#     sum += el
#
# ave = sum/len(result)
#
# print("t =", nx.transitivity(G))
#
# fig = plt.figure("Degree of a random graph", figsize=(8, 8))
# # Create a gridspec for adding subplots of different sizes
# axgrid = fig.add_gridspec(5, 4)
#
# ax0 = fig.add_subplot(axgrid[0:3, :])
# Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
# pos = nx.spring_layout(Gcc, seed=10396953)
# nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
# nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
# ax0.set_title("Connected components of G")
# ax0.set_axis_off()
#
# ax1 = fig.add_subplot(axgrid[3:, :2])
# ax1.plot(degree_sequence, "b-", marker="o")
# ax1.set_title("Degree Rank Plot")
# ax1.set_ylabel("Degree")
# ax1.set_xlabel("Rank")
#
# ax2 = fig.add_subplot(axgrid[3:, 2:])
# ax2.bar(*np.unique(degree_sequence, return_counts=True))
# ax2.set_title("Degree histogram")
# ax2.set_xlabel("Degree")
# ax2.set_ylabel("# of Nodes")
#
# fig.tight_layout()
# plt.show()
#
#
# import numpy as np
# from scipy import stats
# rng = np.random.default_rng()
# rvs = stats.uniform.rvs(size=50, random_state=rng)
#
# rvs = stats.uniform.rvs(size=(100, 50), random_state=rng)
# res = stats.ttest_1samp(rvs, popmean=0.5, axis=1)
# np.sum(res.pvalue < 0.01)
#
