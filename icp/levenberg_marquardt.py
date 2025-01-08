import torch

class LM_optimizer(torch.nn.Module):
    def __init__(self, damping_factor=1e-3):
        super().__init__()
        self.damping_factor = damping_factor

    def forward(self, residuals, Jf):
        valid_mask = ~torch.isnan(Jf)
        print(torch.max(Jf[valid_mask]))
        print("Jf:", Jf.view(-1)[30000:30100])
        print("Jf:", Jf.view(-1)[90000:90100])
        Jt = Jf.transpose(-1, -2)

        Jtj = torch.bmm(Jt, Jf)
        Jtj = Jtj.sum(dim=0)

        Jtr = torch.bmm(Jt, residuals)
        Jtr = Jtr.sum(dim=0)

        diagJtj = torch.diagonal(Jtj)
        epsilon = self.damping_factor * diagJtj
        Hessian = Jtj + torch.diag(epsilon)
        print("Hessian:", Hessian)

        # TODO: Add exception handling for cholesky (torch._C._LinAlgError: linalg.cholesky: The factorization could not be completed because the input is not positive-definite (t)
        # Linear solver for H @ xi = -Jtr
        if Hessian.device.type == 'mps':
            Hessian = Hessian.to("cpu")
            Jtr = Jtr.to("cpu")
            # L = torch.linalg.cholesky(Hessian)
            # delta_parameters = torch.cholesky_solve(-Jtr, L)
            delta_parameters = torch.linalg.solve(Hessian, -Jtr)
            delta_parameters = delta_parameters.to("mps")
        else:
            L = torch.linalg.cholesky(Hessian)
            delta_parameters = torch.cholesky_solve(-Jtr, L)

        return delta_parameters


