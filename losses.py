import torch

# ---------------- MSE ----------------
def recon_loss_fn(recon, x):
    return ((recon - x) ** 2).mean()

# ---------------- MMD ----------------
def mmd_loss(z, prior_z):
    x_kernel = torch.mm(z, z.t())
    y_kernel = torch.mm(prior_z, prior_z.t())
    xy_kernel = torch.mm(z, prior_z.t())

    return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()