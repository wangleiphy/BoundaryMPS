import torch 
from sklearn.decomposition import NMF

def nmf_sklearn(A, k):
    model = NMF(n_components=k, init='random', random_state=0)
    X = model.fit_transform(A.detach().numpy())
    Y = model.components_
    return torch.from_numpy(X).float() , torch.from_numpy(Y).float()

def nmf(A, k):
    m, n = A.shape
    X = torch.rand(m, k)
    Y = torch.rand(k, n)

    for i in range(200):
        X = X*((A@Y.t())/(X@Y@Y.t()))
        Y = Y*((X.t()@A)/(X.t()@X@Y))

    return X, Y 

if __name__=='__main__':
    import numpy as np 
    A = np.array([[1., 1.], [2., 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]], dtype=np.float32)
    A = torch.from_numpy(A)
    X, Y = nmf(A, 2)
    print (X@Y-A)
    
    X, Y = nmf_sklearn(A, 2)
    print (X@Y-A)
