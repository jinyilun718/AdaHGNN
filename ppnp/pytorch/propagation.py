import numpy as np
import scipy.sparse as sp
import torch

class process_graph():
    def __init__(self, adj_matrix, sample_weight=None):
        self.adj_matrix = adj_matrix  # n_feature: C, n_hidden: H
        self.sample_weight = sample_weight
        self.features = None# n_hidden: H, n_hidden: H
        self.A_I = None  # n_hidden: H, n_classes: F
        self.A = None
        self.D_vec = None
        self.E_vec = None
        self.D_invsqrt_corr = None
        self.BH_T = None
        self.DH = None

    def calc_A_hat(self):
        nnodes = self.adj_matrix.shape[0]
        if self.adj_matrix.shape[0] != self.adj_matrix.shape[1]:
            self.A_I = self.adj_matrix.copy()

        else:
            self.A_I = self.adj_matrix + sp.eye(nnodes)  # self-loop

        # paddings_num = self.A_I.shape[0] - self.A_I.shape[1]
        # paddings = np.zeros([self.A_I.shape[0], paddings_num])
        # self.A_I = sp.hstack([self.A_I, paddings])
        self.D_vec = np.sum(self.A_I, axis=1).A1  # calculate node degree
        self.E_vec = np.sum(self.A_I, axis=0).A1  # calculate edge degree
        E_vec_invsqrt_corr = 1 / self.E_vec
        D_vec_invsqrt_corr = 1 / np.sqrt(self.D_vec)
        E_vec_invsqrt_corr[E_vec_invsqrt_corr == float("inf")] = 0
        D_vec_invsqrt_corr[D_vec_invsqrt_corr == float("inf")] = 0

        self.D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)  # vec to matrix
        self.E_invsqrt_corr = sp.diags(E_vec_invsqrt_corr)

        # A = D(-1/2) * H * W * E(-1) * H(T) *D(-1/2)
        self.BH_T = self.E_invsqrt_corr @ self.trans(self.A_I) @ self.D_invsqrt_corr
        self.DH = self.D_invsqrt_corr @ self.A_I
        self.A = self.DH @ self.BH_T
        self.BH_T = torch.FloatTensor(self.BH_T.todense())
        self.DH = torch.FloatTensor(self.DH.todense())

        return self.A

    def calc_simple_hat(self):
        hyper_A = self.adj_matrix
        simple_A = np.zeros([hyper_A.shape[0], hyper_A.shape[0]])
        hyper_edge_nums = hyper_A.shape[1]
        for i in range(hyper_edge_nums):
            edge = hyper_A[:, i]
            non_zeros = edge.nonzero()[0]
            for j in range(len(non_zeros)):
                x_index = []

                x_index.append(non_zeros[j])
                y_index = list(np.delete(non_zeros, j))
                x_index = x_index * len(y_index)
                simple_A[x_index, y_index] = 1.0
        simple_A = simple_A + sp.eye(simple_A.shape[0])
        D_vec = np.sum(simple_A, axis=1).A1
        D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
        D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
        simple_A = D_invsqrt_corr @ simple_A @ D_invsqrt_corr
        return simple_A


    def trans(self, D):
        x = sp.find(D)
        return sp.csr_matrix((x[2], (x[1], x[0])), shape=(D.shape[1], D.shape[0]))

    def calc_hgcn_adj(self, V, E, X, m=True): #Refer paper HyperGCN
        return Laplacian(V, E, X, m)

def Laplacian(V, E, X, m):
    """
    approximates the E defined by the E Laplacian with/without mediators

    arguments:
    V: number of vertices
    E: dictionary of hyperedges (key: hyperedge, value: list/set of hypernodes)
    X: features on the vertices
    m: True gives Laplacian with mediators, while False gives without

    A: adjacency matrix of the graph approximation
    returns:
    updated data with 'graph' as a key and its value the approximated hypergraph
    """

    edges, weights = [], {}
    rv = np.random.rand(X.shape[1])

    for k in E.keys():
        hyperedge = list(E[k])

        p = np.dot(X[hyperedge], rv)  # projection onto a random vector rv
        s, i = np.argmax(p), np.argmin(p)
        Se, Ie = hyperedge[s], hyperedge[i]

        # two stars with mediators
        c = 2 * len(hyperedge) - 3  # normalisation constant

        # <editor-fold desc="加入mediators">
        if m:

            # connect the supremum (Se) with the infimum (Ie)
            edges.extend([[Se, Ie], [Ie, Se]])
            weights[(Se, Ie)] += float(1 / c)

            if (Ie, Se) not in weights:
                weights[(Ie, Se)] = 0
            weights[(Ie, Se)] += float(1 / c)

            # connect the supremum (Se) and the infimum (Ie) with each mediator
            for mediator in hyperedge:
                if mediator != Se and mediator != Ie:
                    edges.extend([[Se, mediator], [Ie, mediator], [mediator, Se], [mediator, Ie]])
                    weights = update(Se, Ie, mediator, weights, c)
            # </editor-fold>
        else:
            edges.extend([[Se, Ie], [Ie, Se]])
            e = len(hyperedge)

            if (Se, Ie) not in weights:
                weights[(Se, Ie)] = 0
            weights[(Se, Ie)] += float(1 / e)

            if (Ie, Se) not in weights:
                weights[(Ie, Se)] = 0
            weights[(Ie, Se)] += float(1 / e)
    A = adjacency(edges, weights, V)
    return A

def adjacency(edges, weights, n):
    """
    computes an sparse adjacency matrix

    arguments:
    edges: list of pairs
    weights: dictionary of edge weights (key: tuple representing edge, value: weight on the edge)
    n: number of nodes

    returns: a scipy.sparse adjacency matrix with unit weight self loops for edges with the given weights
    """
    '''创建dict'''

    dictionary = {tuple(item): index for index, item in enumerate(edges)}
    edges = [list(itm) for itm in dictionary.keys()]
    organised = []

    for e in edges:
        i,j = e[0],e[1]
        w = weights[(i,j)]
        organised.append(w)

    edges, weights = np.array(edges), np.array(organised)
    adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + sp.eye(n)

    A = symnormalise(sp.csr_matrix(adj, dtype=np.float32))
    A = ssm2tst(A)
    return A

def symnormalise(M):
    """
    symmetrically normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1/2} M D^{-1/2}
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    dhi = np.power(d, -1 / 2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)  # D half inverse i.e. D^{-1/2}

    return (DHI.dot(M)).dot(DHI)

def update(Se, Ie, mediator, weights, c):
    """
    updates the weight on {Se,mediator} and {Ie,mediator}
    """

    if (Se, mediator) not in weights:
        weights[(Se, mediator)] = 0
    weights[(Se, mediator)] += float(1 / c)

    if (Ie, mediator) not in weights:
        weights[(Ie, mediator)] = 0
    weights[(Ie, mediator)] += float(1 / c)

    if (mediator, Se) not in weights:
        weights[(mediator, Se)] = 0
    weights[(mediator, Se)] += float(1 / c)

    if (mediator, Ie) not in weights:
        weights[(mediator, Ie)] = 0
    weights[(mediator, Ie)] += float(1 / c)

    return weights

def ssm2tst(M):
    """
    converts a scipy sparse matrix (ssm) to a torch sparse tensor (tst)

    arguments:
    M: scipy sparse matrix

    returns:
    a torch sparse tensor of M
    """

    M = M.tocoo().astype(np.float32)

    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)

    return torch.sparse.FloatTensor(indices, values, shape)




