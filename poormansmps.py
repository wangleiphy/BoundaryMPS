import torch 
import math 
from nmf import nmf_sklearn as nmf 

class MPS(object):
    def __init__(self,L,D=1):
        self.L = L 
        self.bdim = [D for i in range(L-1)]+[1]
        self.tensors = [torch.randn(self.bdim[i-1],2,self.bdim[i]) for i in range(L)]

    def __getitem__(self, i):
        return self.tensors[i]

    def __setitem__(self, i, t):
        self.tensors[i] = t

class MPO(object):
    def __init__(self, L, D=1):
        self.L = L 
        self.bdim=[D for i in range(L-1)]+[1]
        self.tensors = [torch.randn(self.bdim[i-1],2,2,self.bdim[i]) for i in range(L)] # ludr 

    def __getitem__(self, i):
        return self.tensors[i]

    def __setitem__(self, i, t):
        self.tensors[i] = t

def compress_nmf(mps, Dcut):
    '''
    cut a mps up to given bond dimension
    '''
    res = 0.0

    #from left to right, svd 
    for site in range(mps.L-1):
        l=mps.bdim[site-1] # left bond dimension
        r=mps.bdim[site]   # current bond dimension

        A=mps[site].view(l*2,r) # A is a matrix unfolded from the current tensor
        Dnew = min(min(Dcut, l*2), r)
        X, Y = nmf(A, Dnew) # here we intent to do QR = A. However there is no BP, so we do SVD instead 
        sX = X.norm(); sY = Y.norm()
        res = res + torch.log(sX) + torch.log(sY)
        X = X/sX; Y = Y/sY 
        mps[site] = X.view(l,2,-1)
        mps[site+1] = (Y@mps[site+1].view(r,-1)).view(-1,2,mps.bdim[site+1])
        mps.bdim[site] = X.shape[1] 

    return res

def compress(mps, Dcut, epsilon=1E-8):
    '''
    cut a mps up to given bond dimension
    '''
    res = 0.0

    #from left to right, svd 
    for site in range(mps.L-1):
        l=mps.bdim[site-1] # left bond dimension
        r=mps.bdim[site]   # current bond dimension

        A=mps[site].view(l*2,r) # A is a matrix unfolded from the current tensor
        U, S, V = torch.svd(A) # here we intent to do QR = A. However there is no BP, so we do SVD instead 
        Dnew = (S>epsilon).sum().item()
        #print (S[:Dnew])

        R = (V[:, :Dnew]*S[:Dnew]).t()
        s = R.norm()
        res = res + torch.log(s)
        R = R/s # devided by norm
        mps[site] = U[:,:Dnew].view(l,2,Dnew)
        mps[site+1] = (R@mps[site+1].view(r,-1)).view(-1,2,mps.bdim[site+1])
        mps.bdim[site] = Dnew
    
    #print (mps.bdim)
    #from right to left, svd
    for site in reversed(range(1, mps.L)):
        l = mps.bdim[site-1]
        r = mps.bdim[site]

        A = mps[site].view(l, r*2)
        U, S, V = torch.svd(A)
        Dnew = min(Dcut, (S>epsilon).sum().item())
        #print (S[:Dnew])
        mps[site] = V[:, :Dnew].t().view(Dnew,2,-1)
        mps[site-1] = (mps[site-1]@ U[:,:Dnew] *S[:Dnew]).view(-1, 2, Dnew)
        mps.bdim[site-1] = Dnew

    #print(mps.bdim)
    #print ('addition:', res)
    return res

def overlap(mps1, mps2, site=None, op=None):
    '''
    if site and op is present, one sanwitch a site operator between the two mps
    '''
    if site!=0:
        E = mps1[0].view(2, mps1.bdim[0]).t() @ mps2[0].view(2, mps2.bdim[0])
    else:
        E = mps1[0].view(2, mps1.bdim[0]).t() @ op @ mps2[0].view(2, mps2.bdim[0])

    for i in range(1, mps1.L):
        if (site!=i):
            E = torch.einsum('ab,ade,bdf->ef', (E, mps1[i], mps2[i]))
        else:
            E = torch.einsum('ab,ame,mn,bnf->ef', (E, mps1[i], op, mps2[i]))
    return E.sum()

def multiply(mpo, mps):
    for site in range(mps.L):
        mps[site] = torch.einsum('ludr,adb->laurb', (mpo[site], mps[site]))
        mps.bdim[site-1] = mpo.bdim[site-1]*mps.bdim[site-1] # only change left 
        mps[site] = mps[site].contiguous().view(mps.bdim[site-1], 2, -1)
    return None

if __name__=='__main__':
    from copy import copy 
    from math import sqrt 
    L = 16
    mps = MPS(L)
    mps_old = copy(mps)
    compress(mps, 10)
    print( overlap(mps, mps_old)/(sqrt(overlap(mps, mps)) *sqrt(overlap(mps_old, mps_old))) )
    
    mpo = MPO(L)
    multiply(mpo, mps)
