import torch

class LM_optimizer(torch.nn.Module):
    def __init__(self, damping_factor):
        super().__init__()
        self.damping_factor = damping_factor

    def forward(self, residuals, Jf):
        Jt = Jf.transpose(-1, -2)

        Jtj = torch.bmm(Jt, Jf)
        Jtj = Jtj.sum(dim=0)

        Jtr = torch.bmm(Jt, residuals)
        Jtr = Jtr.sum(dim=0)

        diagJtj = torch.diagonal(Jtj)
        epsilon = self.damping_factor * diagJtj
        Hessian = Jtj + torch.diag(epsilon)

        # Linear solver for H @ xi = -Rhs
        L = torch.linalg.cholesky(Hessian)
        delta_parameters = torch.cholesky_solve(-Jtr, L)

        return delta_parameters


