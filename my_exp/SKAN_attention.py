import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SingleHeadAttention, self).__init__()
        self.embed_dim = embed_dim

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        Q = self.q_proj(x)  # [B, T, D]
        K = self.k_proj(x)  # [B, T, D]
        V = self.v_proj(x)  # [B, T, D]

        # Scaled dot-product attention
        d_k = self.embed_dim ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k  # [B, T, T]
        attn_weights = F.softmax(scores, dim=-1)             # [B, T, T]
        attn_output = torch.matmul(attn_weights, V)          # [B, T, D]

        # Output projection
        output = self.out_proj(attn_output)  # [B, T, D]
        return output, attn_weights

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AttentionClassifier(nn.Module):
    def __init__(self, embed_dim, seq_len, num_classes):
        super().__init__()
        self.attn = SingleHeadAttention(embed_dim)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(seq_len * embed_dim, num_classes)

    def forward(self, x):
        # x: [B, seq_len, embed_dim]
        out, attn_weights = self.attn(x)
        out = self.flatten(out)  # Flatten before final layer
        logits = self.classifier(out)
        return logits, attn_weights


def visualize_attention(attn_matrix, input_labels=None, title="Attention Map"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(6, 5))
    sns.heatmap(attn_matrix.detach().numpy(), cmap='viridis',
                xticklabels=False, yticklabels=False,
                square=True, cbar=True, linewidths=0.1)

    plt.title(title)
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.savefig("attention_plot.png")
    plt.close()

batch_size = 2
seq_len = 5
embed_dim = 8

x = torch.randn(batch_size, seq_len, embed_dim)
attention = SingleHeadAttention(embed_dim)
output, weights = attention(x)

print("Output shape:", output.shape)        # [2, 5, 8]
print("Attention weights shape:", weights.shape)  # [2, 5, 5]
print(count_parameters(attention))  # prints 288

import torchvision
from torch.utils.data import DataLoader

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.view(28, 28))  # Treat each row as a timestep
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentionClassifier(embed_dim=28, seq_len=28, num_classes=10).to(device)
print(count_parameters(model))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    total_loss, total_correct = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
    model.eval()
    correct = 0
    total = 0
    all_attn_weights = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            logits, attn_weights = model(x)  # model must return both
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_attn_weights.append(attn_weights.cpu())  # store for visualization

    accuracy = correct / total
    acc = total_correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f} - Train Acc: {acc:.4f} - Test Acc: {accuracy:.4f}")

attn_matrix = all_attn_weights[-1][0]  # shape: [T, T]
visualize_attention(attn_matrix)
'''

#####################################
#SKAN attention
'''
def lrelu(x, k):
    return torch.clamp(k*x, min=0)
class SKANLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, base_function=lrelu, device='cpu'):
        super(SKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.base_function = base_function
        self.device = device
        if bias:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features + 1).to(device))
        else:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        x = x.view(-1, 1, self.in_features)
        # 添加偏置单元
        if self.use_bias:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=2)

        y = self.base_function(x, self.weight)

        y = torch.sum(y, dim=2)
        return y

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

class SKANAttention(nn.Module):
    def __init__(self, embed_dim, base_function=lrelu, device='cpu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device

        self.q_proj = SKANLinear(embed_dim, embed_dim, base_function=base_function, device=device)
        self.k_proj = SKANLinear(embed_dim, embed_dim, base_function=base_function, device=device)
        self.v_proj = SKANLinear(embed_dim, embed_dim, base_function=base_function, device=device)
        self.out_proj = SKANLinear(embed_dim, embed_dim, base_function=base_function, device=device)

    def forward(self, x):
        B, T, E = x.shape
        x_flat = x.view(B * T, E)

        q = self.q_proj(x_flat).view(B, T, E)
        k = self.k_proj(x_flat).view(B, T, E)
        v = self.v_proj(x_flat).view(B, T, E)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (E ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out_flat = out.view(B * T, E)
        out_proj = self.out_proj(out_flat).view(B, T, E)

        return out_proj, attn

class SKANAttentionClassifier(nn.Module):
    def __init__(self, embed_dim, seq_len, num_classes, base_function, device='cpu'):
        super().__init__()
        self.attn = SKANAttention(embed_dim, base_function=base_function, device=device)
        self.flatten = nn.Flatten()
        self.classifier = SKANLinear(seq_len * embed_dim, num_classes, base_function=base_function, device=device)

    def forward(self, x):
        # x: [B, seq_len, embed_dim]
        out, attn_weights = self.attn(x)       # [B, seq_len, embed_dim]
        out = self.flatten(out)                # [B, seq_len * embed_dim]
        logits = self.classifier(out)          # [B, num_classes]
        return logits, attn_weights


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SKANAttentionClassifier(embed_dim=28, seq_len=28, num_classes=10,
                                base_function=lrelu, device=device).to(device)
print(count_parameters(model))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    total_loss, total_correct = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
    model.eval()
    correct = 0
    total = 0
    all_attn_weights = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            logits, attn_weights = model(x)  # model must return both
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_attn_weights.append(attn_weights.cpu())  # store for visualization

    accuracy = correct / total
    acc = total_correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f} - Train Acc: {acc:.4f} - Test Acc: {accuracy:.4f}")

attn_matrix = all_attn_weights[-1][0]  # shape: [T, T]
visualize_attention(attn_matrix)
'''

def lrelu(x, k):
    return torch.clamp(k * x, min=0)

class SKANLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, base_function=lrelu, device='cpu'):
        super(SKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.base_function = base_function
        self.device = device
        if bias:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features + 1).to(device))
        else:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        x = x.view(-1, 1, self.in_features)
        if self.use_bias:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=2)
        y = self.base_function(x, self.weight)
        y = torch.sum(y, dim=2)
        return y
class EfficientSKANAttention(nn.Module):
    def __init__(self, embed_dim, seq_len, base_function=lrelu, device='cpu'):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.attn_score_proj = SKANLinear(embed_dim, seq_len, base_function=base_function, device=device)
        self.value_proj = SKANLinear(embed_dim, embed_dim, base_function=base_function, device=device)

    def forward(self, x):
        B, T, E = x.shape
        assert T == self.seq_len and E == self.embed_dim

        # Compute attention scores directly
        attn_scores = self.attn_score_proj(x.view(B * T, E)).view(B, T, T)
        attn_weights = F.softmax(attn_scores / E ** 0.5, dim=-1)  # [B, T, T]

        # Compute values
        V = self.value_proj(x.view(B * T, E)).view(B, T, E)

        # Attention-weighted sum
        out = torch.bmm(attn_weights, V)  # [B, T, E]

        return out, attn_weights

class EfficientSKANAttentionClassifier(nn.Module):
    def __init__(self, embed_dim, seq_len, num_classes, base_function=lrelu, device='cpu'):
        super().__init__()
        self.attn = EfficientSKANAttention(embed_dim, seq_len, base_function=base_function, device=device)
        self.flatten = nn.Flatten()
        self.classifier = SKANLinear(seq_len * embed_dim, num_classes, base_function=base_function, device=device)

    def forward(self, x):
        # x: [B, T, E]
        out, attn_weights = self.attn(x)
        out = self.flatten(out)
        logits = self.classifier(out)
        return logits, attn_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EfficientSKANAttentionClassifier(embed_dim=28, seq_len=28, num_classes=10, base_function=lrelu, device='cpu')
print(count_parameters(model))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    total_loss, total_correct = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
    model.eval()
    correct = 0
    total = 0
    all_attn_weights = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            logits, attn_weights = model(x)  # model must return both
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_attn_weights.append(attn_weights.cpu())  # store for visualization

    accuracy = correct / total
    acc = total_correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f} - Train Acc: {acc:.4f} - Test Acc: {accuracy:.4f}")

attn_matrix = all_attn_weights[-1][0]  # shape: [T, T]
visualize_attention(attn_matrix)