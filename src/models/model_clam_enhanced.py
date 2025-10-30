import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.model_clam import CLAM_SB, CLAM_MB

# ============================================================================
# MÓDULOS DE REDUCCIÓN DIMENSIONAL
# ============================================================================

class DimensionalityReducer(nn.Module):
    """
    Reduce dimensionalidad para evitar Efecto de Hughes
    en cálculos de atención y distancia
    """
    def __init__(self, input_dim, output_dim, method='linear'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.method = method
        
        if method == 'linear':
            self.projection = nn.Linear(input_dim, output_dim)
        elif method == 'mlp':
            self.projection = nn.Sequential(
                nn.Linear(input_dim, (input_dim + output_dim) // 2),
                nn.ReLU(),
                nn.Linear((input_dim + output_dim) // 2, output_dim)
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def forward(self, x):
        """
        Args:
            x: [N, input_dim]
        Returns:
            x_reduced: [N, output_dim]
        """
        return self.projection(x)


# ============================================================================
# SAMPLERS CON REDUCCIÓN DIMENSIONAL
# ============================================================================

class TopologicalDiversitySampler(nn.Module):
    """
    FPS con reducción dimensional para evitar Efecto de Hughes
    """
    def __init__(self, alpha=0.5, input_dim=None, reduced_dim=None):
        super().__init__()
        self.alpha = alpha
        
        # Si se especifica reducción dimensional
        if input_dim is not None and reduced_dim is not None:
            self.use_reduction = True
            self.reducer = DimensionalityReducer(input_dim, reduced_dim, method='linear')
        else:
            self.use_reduction = False
            self.reducer = None
    
    def forward(self, features, attention_scores, k):
        """
        Args:
            features: [N, D] - vectores originales (alta dimensión)
            attention_scores: [N] - scores de atención
            k: número de patches a seleccionar
        Returns:
            indices: [k] - índices seleccionados
        """
        device = features.device
        N = features.shape[0]
        
        if k >= N:
            return torch.arange(N, device=device)
        
        # Reducir dimensionalidad si está habilitado
        if self.use_reduction:
            features_for_distance = self.reducer(features)
        else:
            features_for_distance = features
        
        # Normalizar features para distancias
        features_norm = F.normalize(features_for_distance, p=2, dim=1)
        
        # Normalizar attention scores
        attention_norm = (attention_scores - attention_scores.min()) / \
                        (attention_scores.max() - attention_scores.min() + 1e-10)
        
        selected_indices = []
        remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
        
        # Primer patch: máximo attention
        first_idx = torch.argmax(attention_scores).item()
        selected_indices.append(first_idx)
        remaining_mask[first_idx] = False
        
        # Distancias mínimas
        min_distances = torch.full((N,), float('inf'), device=device)
        
        # Selección iterativa
        for _ in range(k - 1):
            if not remaining_mask.any():
                break
            
            # Distancia al último seleccionado
            last_selected_feat = features_norm[selected_indices[-1]].unsqueeze(0)
            remaining_feats = features_norm[remaining_mask]
            distances = torch.cdist(remaining_feats, last_selected_feat, p=2).squeeze(1)
            
            remaining_indices = torch.where(remaining_mask)[0]
            min_distances[remaining_mask] = torch.min(
                min_distances[remaining_mask], 
                distances
            )
            
            # Score combinado
            combined_scores = (1 - self.alpha) * attention_norm[remaining_mask] + \
                             self.alpha * min_distances[remaining_mask]
            
            best_idx_in_remaining = torch.argmax(combined_scores).item()
            best_idx = remaining_indices[best_idx_in_remaining].item()
            
            selected_indices.append(best_idx)
            remaining_mask[best_idx] = False
        
        return torch.tensor(selected_indices, dtype=torch.long, device=device)


class DPPSampler(nn.Module):
    """
    DPP con reducción dimensional
    """
    def __init__(self, lambda_reg=0.01, input_dim=None, reduced_dim=None):
        super().__init__()
        self.lambda_reg = lambda_reg
        
        if input_dim is not None and reduced_dim is not None:
            self.use_reduction = True
            self.reducer = DimensionalityReducer(input_dim, reduced_dim, method='linear')
        else:
            self.use_reduction = False
            self.reducer = None
    
    def compute_kernel_matrix(self, features, attention_scores):
        N = features.shape[0]
        device = features.device
        
        # Reducir dimensionalidad
        if self.use_reduction:
            features = self.reducer(features)
        
        features_norm = F.normalize(features, p=2, dim=1)
        S = torch.mm(features_norm, features_norm.t())
        
        q = attention_scores.unsqueeze(1)
        L = torch.sqrt(torch.mm(q, q.t())) * S
        L = L + self.lambda_reg * torch.eye(N, device=device)
        
        return L
    
    def greedy_map_inference(self, L, k):
        N = L.shape[0]
        device = L.device
        
        selected_indices = []
        remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
        
        for _ in range(k):
            if not remaining_mask.any():
                break
            
            best_score = -float('inf')
            best_idx = None
            
            for idx in torch.where(remaining_mask)[0]:
                test_indices = selected_indices + [idx.item()]
                L_Y = L[test_indices][:, test_indices]
                
                sign, logdet = torch.linalg.slogdet(L_Y)
                score = logdet.item() if sign > 0 else -float('inf')
                
                if score > best_score:
                    best_score = score
                    best_idx = idx.item()
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_mask[best_idx] = False
        
        return torch.tensor(selected_indices, dtype=torch.long, device=device)
    
    def forward(self, features, attention_scores, k):
        L = self.compute_kernel_matrix(features, attention_scores)
        return self.greedy_map_inference(L, k)


class MaxMinSampler(nn.Module):
    """
    Max-Min con reducción dimensional
    """
    def __init__(self, input_dim=None, reduced_dim=None):
        super().__init__()
        
        if input_dim is not None and reduced_dim is not None:
            self.use_reduction = True
            self.reducer = DimensionalityReducer(input_dim, reduced_dim, method='linear')
        else:
            self.use_reduction = False
            self.reducer = None
    
    def forward(self, features, attention_scores, k):
        device = features.device
        N = features.shape[0]
        
        if k >= N:
            return torch.arange(N, device=device)
        
        # Reducir dimensionalidad
        if self.use_reduction:
            features = self.reducer(features)
        
        features_norm = F.normalize(features, p=2, dim=1)
        
        selected_indices = []
        first_idx = torch.argmax(attention_scores).item()
        selected_indices.append(first_idx)
        
        selected_feat = features_norm[first_idx].unsqueeze(0)
        min_distances = torch.cdist(features_norm, selected_feat, p=2).squeeze(1)
        min_distances[first_idx] = -float('inf')
        
        for _ in range(k - 1):
            farthest_idx = torch.argmax(min_distances).item()
            selected_indices.append(farthest_idx)
            
            new_feat = features_norm[farthest_idx].unsqueeze(0)
            new_distances = torch.cdist(features_norm, new_feat, p=2).squeeze(1)
            min_distances = torch.min(min_distances, new_distances)
            min_distances[farthest_idx] = -float('inf')
        
        return torch.tensor(selected_indices, dtype=torch.long, device=device)


# ============================================================================
# CLAM CON ATENCIÓN EN ESPACIO REDUCIDO
# ============================================================================

class CLAM_SB_Enhanced(CLAM_SB):
    """
    CLAM_SB que calcula atención y diversidad en espacio reducido
    pero agrega features originales para clasificación
    """
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8,
                 n_classes=2, instance_loss_fn=nn.CrossEntropyLoss(),
                 subtyping=False, embed_dim=1024,
                 # Parámetros de reducción dimensional
                 reduced_dim=256,  # Dimensión reducida para atención/distancia
                 diversity_method='fps',
                 diversity_alpha=0.5):
        
        # Guardar dimensiones
        self.original_embed_dim = embed_dim
        self.reduced_dim = reduced_dim
        
        # Inicializar CLAM_SB con embed_dim ORIGINAL
        super(CLAM_SB_ReducedSpace, self).__init__(
            gate=gate,
            size_arg=size_arg,
            dropout=dropout,
            k_sample=k_sample,
            n_classes=n_classes,
            instance_loss_fn=instance_loss_fn,
            subtyping=subtyping,
            embed_dim=embed_dim  # Usa dimensión original para agregación
        )
        
        # Reductor dimensional para atención
        self.attention_reducer = DimensionalityReducer(
            embed_dim, reduced_dim, method='linear'
        )
        
        # Sampler topológico con reducción
        if diversity_method == 'fps':
            self.topological_sampler = TopologicalDiversitySampler(
                alpha=diversity_alpha,
                input_dim=embed_dim,
                reduced_dim=reduced_dim
            )
        elif diversity_method == 'dpp':
            self.topological_sampler = DPPSampler(
                lambda_reg=0.01,
                input_dim=embed_dim,
                reduced_dim=reduced_dim
            )
        elif diversity_method == 'maxmin':
            self.topological_sampler = MaxMinSampler(
                input_dim=embed_dim,
                reduced_dim=reduced_dim
            )
        else:
            raise ValueError(f"Unknown diversity_method: {diversity_method}")
        
        self.diversity_method = diversity_method
    
    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        """
        Forward modificado:
        1. Reducir dimensión para calcular atención
        2. Usar embeddings originales para agregación
        """
        # Reducir dimensionalidad para calcular atención
        h_reduced = self.attention_reducer(h)  # [N, reduced_dim]
        
        # Calcular atención en espacio reducido
        A, _ = self.attention_net(h_reduced)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        
        if attention_only:
            return A
        
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                
                if inst_label == 1:  # in-the-class
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
        
        # Agregación con features ORIGINALES
        M = torch.mm(A, h)  # Usa h original, no h_reduced
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        
        if instance_eval:
            results_dict = {
                'instance_loss': total_inst_loss,
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_preds)
            }
        else:
            results_dict = {}
        
        if return_features:
            results_dict.update({'features': M})
        
        return logits, Y_prob, Y_hat, A_raw, results_dict
    
    def inst_eval(self, A, h, classifier):
        """
        Evaluación de instancia usando diversidad topológica
        h son las features ORIGINALES (no reducidas)
        """
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        
        A_flat = A.view(-1)
        
        # Sampler usa h original pero calcula distancias en espacio reducido internamente
        top_p_ids = self.topological_sampler(h, A_flat, self.k_sample)
        top_p = torch.index_select(h, dim=0, index=top_p_ids)  # Features originales
        
        top_n_ids = self.topological_sampler(h, -A_flat, self.k_sample)
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)
        
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)  # Features originales
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        
        return instance_loss, all_preds, all_targets
    
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        
        A_flat = A.view(-1)
        
        top_p_ids = self.topological_sampler(h, A_flat, self.k_sample)
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        
        return instance_loss, p_preds, p_targets


class CLAM_MB_Enhanced(CLAM_MB):
    """
    CLAM_MB con atención y diversidad en espacio reducido
    """
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8,
                 n_classes=2, instance_loss_fn=nn.CrossEntropyLoss(),
                 subtyping=False, embed_dim=1024,
                 reduced_dim=256,
                 diversity_method='fps',
                 diversity_alpha=0.5):
        
        self.original_embed_dim = embed_dim
        self.reduced_dim = reduced_dim
        
        super(CLAM_MB_ReducedSpace, self).__init__(
            gate=gate,
            size_arg=size_arg,
            dropout=dropout,
            k_sample=k_sample,
            n_classes=n_classes,
            instance_loss_fn=instance_loss_fn,
            subtyping=subtyping,
            embed_dim=embed_dim
        )
        
        # Reductor dimensional
        self.attention_reducer = DimensionalityReducer(
            embed_dim, reduced_dim, method='linear'
        )
        
        # Sampler topológico
        if diversity_method == 'fps':
            self.topological_sampler = TopologicalDiversitySampler(
                alpha=diversity_alpha,
                input_dim=embed_dim,
                reduced_dim=reduced_dim
            )
        elif diversity_method == 'dpp':
            self.topological_sampler = DPPSampler(
                lambda_reg=0.01,
                input_dim=embed_dim,
                reduced_dim=reduced_dim
            )
        elif diversity_method == 'maxmin':
            self.topological_sampler = MaxMinSampler(
                input_dim=embed_dim,
                reduced_dim=reduced_dim
            )
        else:
            raise ValueError(f"Unknown diversity_method: {diversity_method}")
        
        self.diversity_method = diversity_method
    
    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        # Reducir para atención
        h_reduced = self.attention_reducer(h)
        
        # Atención en espacio reducido
        A, _ = self.attention_net(h_reduced)
        A = torch.transpose(A, 1, 0)
        
        if attention_only:
            return A
        
        A_raw = A
        A = F.softmax(A, dim=1)

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                
                if inst_label == 1:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        # Agregación con features originales
        M = torch.mm(A, h)

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        
        if instance_eval:
            results_dict = {
                'instance_loss': total_inst_loss,
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_preds)
            }
        else:
            results_dict = {}
        
        if return_features:
            results_dict.update({'features': M})
        
        return logits, Y_prob, Y_hat, A_raw, results_dict
    
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        
        A_flat = A.view(-1)
        
        top_p_ids = self.topological_sampler(h, A_flat, self.k_sample)
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        
        top_n_ids = self.topological_sampler(h, -A_flat, self.k_sample)
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)
        
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        
        return instance_loss, all_preds, all_targets
    
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        
        A_flat = A.view(-1)
        
        top_p_ids = self.topological_sampler(h, A_flat, self.k_sample)
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        
        return instance_loss, p_preds, p_targets