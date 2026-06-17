import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Cross-View Contrastive Loss (Section 4.4, Eqs. 18-22)
# ============================================================
# - Projection head (line 18-22): Eq. 18
# - Cosine similarity + temperature (line 29-35): exp(sim/τ) in Eq. 20-21
# - Positive sample matrix 'pos' (external input): Eq. 19 with threshold θ_pos
# - Loss computation (line 43-47): Eq. 20 (ns->mh) and Eq. 21 (mh->ns)
# - Weighted combination (line 48): Eq. 22
# ============================================================

class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mh, z_ns, pos):
        z_proj_mh = self.proj(z_mh)
        z_proj_ns = self.proj(z_ns)
        matrix_mh2ns = self.sim(z_proj_mh, z_proj_ns)
        matrix_ns2mh = matrix_mh2ns.t()

        matrix_mh2ns = matrix_mh2ns/(torch.sum(matrix_mh2ns, dim=1).view(-1, 1) + 1e-8)
        lori_mh = -torch.log(matrix_mh2ns.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_ns2mh = matrix_ns2mh / (torch.sum(matrix_ns2mh, dim=1).view(-1, 1) + 1e-8)
        lori_ns = -torch.log(matrix_ns2mh.mul(pos.to_dense()).sum(dim=-1)).mean()
        return self.lam * lori_mh + (1 - self.lam) * lori_ns


