"""
Multi-Level CLAM (MCLAM)
Hierarchical attention-based MIL with spatial refinement.

The algorithm works in multiple levels:
1. First level: Select top-K most attentive instances
2. Create bounding box covering selected instances
3. Filter instances within the bounding box
4. Second level: Apply attention on filtered region
5. Select top-M instances from refined region
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn_Net_Gated(nn.Module):
    """Gated Attention Network (from CLAM)"""
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class MCLAM(nn.Module):
    """
    Multi-Level CLAM: Hierarchical attention with spatial refinement

    Args:
        n_classes: Number of output classes
        in_dim: Input feature dimension
        hidden_dim: Hidden dimension for attention networks
        dropout: Whether to use dropout
        k_sample_level1: Number of instances to select in first level
        k_sample_level2: Number of instances to select in second level
        size_arg: Size configuration ('small' or 'big')
        subtyping: Whether this is a subtyping problem
    """
    def __init__(self, n_classes=2, in_dim=1024, hidden_dim=256, dropout=True,
                 k_sample_level1=8, k_sample_level2=8, size_arg='small',
                 subtyping=False, instance_loss_fn=None):
        super(MCLAM, self).__init__()

        self.n_classes = n_classes
        self.subtyping = subtyping
        self.k_sample_level1 = k_sample_level1
        self.k_sample_level2 = k_sample_level2

        # Size configurations
        size = {'small': [in_dim, hidden_dim, hidden_dim],
                'big': [in_dim, 512, hidden_dim]}[size_arg]

        # Feature projection
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        # Level 1 attention
        if self.subtyping:
            fc.append(nn.Linear(size[1], size[1]))
            fc.append(nn.ReLU())
            if dropout:
                fc.append(nn.Dropout(0.25))

        self.fc = nn.Sequential(*fc)

        # Level 1 and Level 2 attention networks
        self.attention_net_level1 = Attn_Net_Gated(L=size[1], D=size[2],
                                                     dropout=dropout, n_classes=1)
        self.attention_net_level2 = Attn_Net_Gated(L=size[1], D=size[2],
                                                     dropout=dropout, n_classes=1)

        # Bag classifier
        bag_classifiers = [nn.Linear(size[1], 1) for _ in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)

        # Instance classifier (if using instance loss)
        self.instance_loss_fn = instance_loss_fn
        if instance_loss_fn is not None:
            instance_classifiers = [nn.Linear(size[1], 2) for _ in range(n_classes)]
            self.instance_classifiers = nn.ModuleList(instance_classifiers)

        self.size = size

    def relocate(self):
        """Move model to GPU if available"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net_level1 = self.attention_net_level1.to(device)
        self.attention_net_level2 = self.attention_net_level2.to(device)
        self.fc = self.fc.to(device)
        self.classifiers = self.classifiers.to(device)
        if self.instance_loss_fn is not None:
            self.instance_classifiers = self.instance_classifiers.to(device)

    def get_spatial_bounding_box(self, coords, top_indices):
        """
        Compute bounding box covering selected instances.

        Args:
            coords: Tensor of shape (N, 2) with (x, y) coordinates
            top_indices: Indices of selected instances

        Returns:
            Tuple of (x_min, x_max, y_min, y_max)
        """
        selected_coords = coords[top_indices]
        x_min = selected_coords[:, 0].min()
        x_max = selected_coords[:, 0].max()
        y_min = selected_coords[:, 1].min()
        y_max = selected_coords[:, 1].max()

        return x_min, x_max, y_min, y_max

    def filter_by_bounding_box(self, features, coords, bbox):
        """
        Filter instances within bounding box.

        Args:
            features: Feature tensor (N, D)
            coords: Coordinate tensor (N, 2)
            bbox: Tuple (x_min, x_max, y_min, y_max)

        Returns:
            Filtered features and their indices
        """
        x_min, x_max, y_min, y_max = bbox

        mask = (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) & \
               (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max)

        filtered_features = features[mask]
        filtered_indices = torch.nonzero(mask).squeeze()

        return filtered_features, filtered_indices

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def inst_eval(self, A, h, classifier):
        """Instance-level evaluation"""
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample_level1)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample_level1, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample_level1, device)
        n_targets = self.create_negative_targets(self.k_sample_level1, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def inst_eval_out(self, A, h, classifier):
        """Instance evaluation for output"""
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample_level1)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_positive_targets(self.k_sample_level1, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, coords=None, label=None, instance_eval=False, return_features=False, attention_only=False):
        """
        Forward pass with hierarchical attention.

        Args:
            h: Input features (1, N, D) or (N, D)
            coords: Spatial coordinates (N, 2) - REQUIRED for spatial refinement
            label: Ground truth label (for instance evaluation)
            instance_eval: Whether to perform instance-level evaluation
            return_features: Whether to return intermediate features
            attention_only: Whether to return only attention scores

        Returns:
            logits: Classification logits
            Y_prob: Class probabilities
            Y_hat: Predicted class
            A_raw: Attention scores (level 1 and level 2)
            results_dict: Additional results
        """
        device = h.device

        # Handle batch dimension
        if len(h.shape) == 3:
            h = h.squeeze(0)

        # Project features
        h = self.fc(h)  # (N, hidden_dim)

        # LEVEL 1: First attention pass
        A_level1, h_level1 = self.attention_net_level1(h)  # A: (N, 1), h: (N, hidden_dim)
        A_level1 = torch.transpose(A_level1, 1, 0)  # (1, N)

        if attention_only:
            return A_level1

        # Get top-K instances from level 1
        A_level1_softmax = F.softmax(A_level1, dim=1)
        top_k_indices_level1 = torch.topk(A_level1_softmax, self.k_sample_level1, dim=1)[1].squeeze()

        # LEVEL 2: Spatial refinement
        if coords is not None and len(coords) == len(h):
            # Create bounding box from top-K instances
            bbox = self.get_spatial_bounding_box(coords, top_k_indices_level1)

            # Filter features within bounding box
            h_filtered, filtered_indices = self.filter_by_bounding_box(h, coords, bbox)

            if len(h_filtered) > 0:
                # Second attention pass on filtered region
                A_level2, h_level2 = self.attention_net_level2(h_filtered)
                A_level2 = torch.transpose(A_level2, 1, 0)

                # Get top-M instances from level 2
                A_level2_softmax = F.softmax(A_level2, dim=1)

                # Use level 2 attention for final aggregation
                M = A_level2_softmax @ h_filtered
            else:
                # Fallback if no instances in bounding box
                M = A_level1_softmax @ h
                A_level2 = A_level1
        else:
            # Fallback if no coordinates provided
            M = A_level1_softmax @ h
            A_level2 = A_level1

        # Classification
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M)

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        # Instance evaluation
        results_dict = {}
        if instance_eval and self.instance_loss_fn is not None:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []

            for c in range(self.n_classes):
                inst_loss, preds, targets = self.inst_eval(A_level1, h, self.instance_classifiers[c])
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                total_inst_loss += inst_loss

            results_dict['instance_loss'] = total_inst_loss / self.n_classes
            results_dict['inst_labels'] = np.array(all_targets)
            results_dict['inst_preds'] = np.array(all_preds)

        if return_features:
            results_dict['features'] = M

        return logits, Y_prob, Y_hat, (A_level1, A_level2), results_dict


if __name__ == '__main__':
    import numpy as np

    # Test the model
    model = MCLAM(n_classes=2, in_dim=512, hidden_dim=256,
                  k_sample_level1=8, k_sample_level2=8)

    # Create dummy data
    n_instances = 1000
    features = torch.randn(1, n_instances, 512)
    coords = torch.rand(n_instances, 2) * 100  # Random coordinates in [0, 100]

    # Forward pass
    logits, Y_prob, Y_hat, (A_level1, A_level2), results_dict = model(features, coords)

    print(f"Input shape: {features.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions: {Y_hat}")
    print(f"Probabilities: {Y_prob}")
    print(f"Level 1 attention shape: {A_level1.shape}")
    print(f"Level 2 attention shape: {A_level2.shape}")
