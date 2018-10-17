import torch 

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

def compress(mps, Dcut):
    '''
    cut a mps up to given bond dimension
    '''
    
    #from left to right, QR 
    for site in range(mps.L-1):
        l=mps.bdim[site-1] # left bond dimension
        r=mps.bdim[site]   # current bond dimension

        A=mps[site].view(l*2,r) # A is a matrix unfolded from the current tensor
        Q,R=torch.qr(A)
        R/=R.norm() # devided by norm
        mps[site] = Q.contiguous().view(l,2,-1)
        mps[site+1] = (R@mps[site+1].view(r,-1)).view(-1,2,mps.bdim[site+1])
        mps.bdim[site] = Q.shape[1] 
    
    print (mps.bdim)
    #from right to left, svd
    for site in reversed(range(1, mps.L)):
        l = mps.bdim[site-1]
        r = mps.bdim[site]

        A = mps[site].view(l, r*2)
        U, S, V = torch.svd(A)
        Dnew = min(Dcut, S.shape[0])
        mps[site] = V[:, :Dnew].t().view(Dnew,2,-1)
        mps[site-1] = (mps[site-1]@ U[:,:Dnew] *S[:Dnew]).view(-1, 2, Dnew)
        mps.bdim[site-1] = Dnew

    print(mps.bdim)
    return None 

def overlap(mps1, mps2):
    E = mps1[0].view(2, mps1.bdim[0]).t() @ mps2[0].view(2, mps2.bdim[0])
    for site in range(1, mps1.L):
        E = torch.einsum('ab,ade,bdf->ef', E, mps1[site], mps2[site])
    return E.item()

def multiply(mpo, mps):
    for site in range(mps.L):
        mps[site] = torch.einsum('ludr,adb->laurb', mpo[site], mps[site])
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
