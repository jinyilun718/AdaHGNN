import numpy as np
adj = [[1,1,1,1,1],
       [1,1,1,1,1],
       [1,1,1,1,0],
       [1,1,1,1,0],
       [1,1,0,0,1]]

idj = [[1,1,1],
       [1,1,1],
       [1,1,0],
       [0,1,0],
         [0,0,1]]

adj = np.array(adj)
idj = np.array(idj)
D = np.diag(np.sum(adj, axis=1))
D = np.linalg.inv(np.sqrt(D))
adj = D@adj@D
DH= np.diag(np.sum(idj, axis=1))
DH = np.linalg.inv(DH)
B = np.diag(np.sum(idj, axis=0))
B = np.linalg.inv(B)
a = DH@idj@B@idj.T
# a = idj@idj.T
# a = np.where(a>0, 1, 0)
print(a)
print(adj)
# print(a==adj)