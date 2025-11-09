import torch.nn as nn
import torch.nn.functional as F
import torch


class ABMIL(nn.Module):
    def __init__(self, n_classes = 2, in_dim=512, hidden_dim=512, dropout=0.3, is_norm=True, *args, **kwargs):
        super().__init__()
        self._fc1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid()
        )
        self.attention_weight = nn.Linear(hidden_dim // 2, 1)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.is_norm = is_norm
    def forward(self, x):
        # Handle both (N, D) and (B, N, D) inputs
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension: (N, D) -> (1, N, D)

        x = self._fc1(x)
        A_U = self.attention_U(x)
        A_V = self.attention_V(x)
        A = self.attention_weight(A_U * A_V).transpose(1, 2)
        if self.is_norm:
            A = torch.softmax(A, dim=-1)
        x = torch.bmm(A, x).squeeze(dim=1)
        x = self.dropout(x)
        logits = self.classifier(x)

        # Ensure logits has batch dimension
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {}
        return logits, Y_prob, Y_hat, None, results_dict
if __name__ == '__main__':
    model = ABMIL()
    data = torch.rand((1, 1000, 512))
    out = model(data)
    print(out)