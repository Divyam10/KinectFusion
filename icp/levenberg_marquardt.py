import torch

class LM_optimizer(torch.nn.Module):
    def __init__(self, max_iterations, damping_factor=1e-3):
        super().__init__()
        self.max_iterations = max_iterations
        self.damping_factor = damping_factor

    def forward(self, residuals, Jf):
        Jt = Jf.transpose(-1, -2)

        Jtj = torch.bmm(Jt, Jf)
        Jtj = Jtj.sum(dim=0)

        Jtj = Jtj.to("cpu")
        det = torch.linalg.det(Jtj)
        print("Jtj determinant:", det)
        Jtj = Jtj.to(Jt.device)

        Jtr = torch.bmm(Jt, residuals)
        Jtr = Jtr.sum(dim=0)

        diagJtj = torch.diagonal(Jtj)
        epsilon = self.damping_factor * diagJtj
        Hessian = Jtj + torch.diag(epsilon)

        # TODO: Add exception handling for cholesky (torch._C._LinAlgError: linalg.cholesky: The factorization could not be completed because the input is not positive-definite (t)
        # Linear solver for H @ del(xi) = -Jtr
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

        err_msg = ""
        if det > 1.e28 or torch.isnan(det):
            err_msg += f"Jtj determinant it too high or NaN - {det}\n"
        return delta_parameters, err_msg


