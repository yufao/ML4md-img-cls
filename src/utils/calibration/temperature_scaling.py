import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))  # log temperature; T=exp(log_t)

    def forward(self, logits):
        T = torch.exp(self.log_t)
        return logits / T

    @torch.no_grad()
    def temperature(self) -> float:
        return float(torch.exp(self.log_t).item())


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> TemperatureScaler:
    scaler = TemperatureScaler().to(logits.device)
    optimizer = optim.LBFGS([scaler.log_t], lr=0.5, max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler
