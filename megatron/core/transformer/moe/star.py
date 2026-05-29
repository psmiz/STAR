import copy
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GHA(nn.Module):
    def __init__(self, dim, n_components, learning_rate=1e-1):
        super().__init__()
        self.dim = dim
        self.n_components = n_components
        self.lr = learning_rate
        self.register_buffer(
            'basis',
            F.normalize(torch.randn(n_components, dim), dim=1)
        )
    
    def components_(self):
        return self.basis.detach()

    def partial_fit(self, X, iters=10):
        # X: (T, B, D) -> flatten to (T*B, D)
        X = X.clone().detach().reshape(-1, X.size(-1))  # (N, D) with N = T*B

        for _ in range(iters):
            # Y: (N, K)
            Y = X @ self.basis.T
            # proj: (N, K, D)
            proj = torch.cumsum(Y[:, :, None] * self.basis[None, :, :], dim=1)
            # delta: (K, D)
            delta = (Y[:, :, None] * (X[:, None, :] - proj)).mean(dim=0)

            with torch.no_grad():
                self.basis.add_(self.lr * delta)
                self.basis.copy_(F.normalize(self.basis, dim=1))

rank = 0

class STARGate(nn.Module):
    def __init__(self, model_dim, num_global_experts, glr=2e-5, **options):
        super().__init__()
        self.dim = model_dim
        self.K = num_global_experts
        self.glr = glr
        self.mixing_coef = nn.Parameter(torch.rand(self.K, self.K), requires_grad=True)
        nn.init.orthogonal_(self.mixing_coef)
        self.gha = GHA(self.dim, self.K, learning_rate=self.glr)
        self.alpha = nn.Parameter(torch.zeros(self.K))

    def update(self, x):
        if self.training:
            self.gha.partial_fit(x, iters=1)
            
    def basismix(self):
        basis = self.gha.components_() 
        mbasis = self.mixing_coef @ basis 
        return mbasis

    def forward(self, x):
        self.update(x)  
        rbasis = self.basismix()
        
        adaptive_logits = torch.matmul(x, rbasis.T)
        alpha = torch.sigmoid(self.alpha)
        
        return adaptive_logits, alpha
    
Gate = STARGate