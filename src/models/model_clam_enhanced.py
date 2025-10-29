import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.model_clam import CLAM_MB, CLAM_SB

# ============================================================================
# SAMPLERS TOPOLÓGICOS (módulos independientes)
# ============================================================================

class TopologicalDiversitySampler(nn.Module):
    """
    Farthest Point Sampling (FPS) con balance attention-diversidad
    Maximiza distancia topológica en el espacio de embeddings
    """
    def __init__(self, alpha=0.5):
        """
        Args:
            alpha: balance entre attention (0) y diversidad topológica (1)
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(self, features, attention_scores, k):
        """
        Args:
            features: [N, D] - vectores de características
            attention_scores: [N] - scores de atención
            k: número de patches a seleccionar
        Returns:
            indices: [k] - índices seleccionados
        """
        device = features.device
        N = features.shape[0]
        
        if k >= N:
            return torch.arange(N, device=device)
        
        # Normalizar features para distancias euclidianas
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Normalizar attention scores
        attention_norm = (attention_scores - attention_scores.min()) / \
                        (attention_scores.max() - attention_scores.min() + 1e-10)
        
        selected_indices = []
        remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
        
        # Inicializar con el patch de mayor attention
        first_idx = torch.argmax(attention_scores).item()
        selected_indices.append(first_idx)
        remaining_mask[first_idx] = False
        
        # Distancias mínimas de cada punto al conjunto seleccionado
        min_distances = torch.full((N,), float('inf'), device=device)
        
        # Iterativamente seleccionar patches
        for _ in range(k - 1):
            if not remaining_mask.any():
                break
            
            # Actualizar distancias mínimas al último punto agregado
            last_selected_feat = features_norm[selected_indices[-1]].unsqueeze(0)
            remaining_feats = features_norm[remaining_mask]
            
            # Distancias euclidianas al último seleccionado
            distances = torch.cdist(remaining_feats, last_selected_feat, p=2).squeeze(1)
            
            # Actualizar distancias mínimas
            remaining_indices = torch.where(remaining_mask)[0]
            min_distances[remaining_mask] = torch.min(
                min_distances[remaining_mask], 
                distances
            )
            
            # Score combinado: (1-alpha) * attention + alpha * min_distance
            combined_scores = (1 - self.alpha) * attention_norm[remaining_mask] + \
                             self.alpha * min_distances[remaining_mask]
            
            # Seleccionar el mejor
            best_idx_in_remaining = torch.argmax(combined_scores).item()
            best_idx = remaining_indices[best_idx_in_remaining].item()
            
            selected_indices.append(best_idx)
            remaining_mask[best_idx] = False
        
        return torch.tensor(selected_indices, dtype=torch.long, device=device)


class DPPSampler(nn.Module):
    """
    Determinantal Point Process (DPP) Sampling
    Maximiza el determinante para obtener diversidad topológica
    """
    def __init__(self, lambda_reg=0.01):
        super().__init__()
        self.lambda_reg = lambda_reg
    
    def compute_kernel_matrix(self, features, attention_scores):
        """
        Construye matriz de kernel L = q_i * q_j * S_ij
        """
        N = features.shape[0]
        device = features.device
        
        # Normalizar features
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Matriz de similitud (cosine similarity)
        S = torch.mm(features_norm, features_norm.t())
        
        # Quality scores (attention)
        q = attention_scores.unsqueeze(1)
        
        # Kernel matrix
        L = torch.sqrt(torch.mm(q, q.t())) * S
        L = L + self.lambda_reg * torch.eye(N, device=device)
        
        return L
    
    def greedy_map_inference(self, L, k):
        """
        Greedy MAP inference: maximiza log det(L_Y)
        """
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
    Max-Min distance sampling
    En cada paso selecciona el patch más lejano del conjunto
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, features, attention_scores, k):
        device = features.device
        N = features.shape[0]
        
        if k >= N:
            return torch.arange(N, device=device)
        
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Inicializar con máximo attention
        selected_indices = []
        first_idx = torch.argmax(attention_scores).item()
        selected_indices.append(first_idx)
        
        # Distancias mínimas
        selected_feat = features_norm[first_idx].unsqueeze(0)
        min_distances = torch.cdist(features_norm, selected_feat, p=2).squeeze(1)
        min_distances[first_idx] = -float('inf')
        
        # Seleccionar patches más lejanos
        for _ in range(k - 1):
            farthest_idx = torch.argmax(min_distances).item()
            selected_indices.append(farthest_idx)
            
            new_feat = features_norm[farthest_idx].unsqueeze(0)
            new_distances = torch.cdist(features_norm, new_feat, p=2).squeeze(1)
            min_distances = torch.min(min_distances, new_distances)
            min_distances[farthest_idx] = -float('inf')
        
        return torch.tensor(selected_indices, dtype=torch.long, device=device)


# ============================================================================
# EXTENSIONES DE CLAM_SB CON DIVERSIDAD TOPOLÓGICA
# ============================================================================

class CLAM_SB_Topological(CLAM_SB):
    """
    Extensión de CLAM_SB que solo modifica el muestreo de patches
    para incluir diversidad topológica en el espacio de embeddings
    """
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, 
                 n_classes=2, instance_loss_fn=nn.CrossEntropyLoss(), 
                 subtyping=False, embed_dim=1024,
                 # Nuevos parámetros
                 diversity_method='fps',  # 'fps', 'dpp', 'maxmin'
                 diversity_alpha=0.5):     # balance attention-diversity (solo para fps)
        
        # Inicializar CLAM_SB original
        super(CLAM_SB_Topological, self).__init__(
            gate=gate, 
            size_arg=size_arg, 
            dropout=dropout, 
            k_sample=k_sample,
            n_classes=n_classes, 
            instance_loss_fn=instance_loss_fn, 
            subtyping=subtyping, 
            embed_dim=embed_dim
        )
        
        # Agregar sampler topológico
        if diversity_method == 'fps':
            self.topological_sampler = TopologicalDiversitySampler(alpha=diversity_alpha)
        elif diversity_method == 'dpp':
            self.topological_sampler = DPPSampler(lambda_reg=0.01)
        elif diversity_method == 'maxmin':
            self.topological_sampler = MaxMinSampler()
        else:
            raise ValueError(f"Unknown diversity_method: {diversity_method}")
        
        self.diversity_method = diversity_method
    
    def inst_eval(self, A, h, classifier):
        """
        Sobrescribe inst_eval para usar diversidad topológica
        """
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        
        A_flat = A.view(-1)
        
        # Seleccionar top patches con diversidad topológica
        top_p_ids = self.topological_sampler(h, A_flat, self.k_sample)
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        
        # Seleccionar patches negativos (invertir scores)
        top_n_ids = self.topological_sampler(h, -A_flat, self.k_sample)
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        
        # Resto igual que CLAM_SB original
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)
        
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        
        return instance_loss, all_preds, all_targets
    
    def inst_eval_out(self, A, h, classifier):
        """
        Sobrescribe inst_eval_out para usar diversidad topológica
        """
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        
        A_flat = A.view(-1)
        
        # Seleccionar con diversidad topológica
        top_p_ids = self.topological_sampler(h, A_flat, self.k_sample)
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        
        return instance_loss, p_preds, p_targets


class CLAM_MB_Topological(CLAM_MB):
    """
    Extensión de CLAM_MB con diversidad topológica
    """
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8,
                 n_classes=2, instance_loss_fn=nn.CrossEntropyLoss(), 
                 subtyping=False, embed_dim=1024,
                 # Nuevos parámetros
                 diversity_method='fps',
                 diversity_alpha=0.5):
        
        # Inicializar CLAM_MB original
        super(CLAM_MB_Topological, self).__init__(
            gate=gate, 
            size_arg=size_arg, 
            dropout=dropout, 
            k_sample=k_sample,
            n_classes=n_classes, 
            instance_loss_fn=instance_loss_fn, 
            subtyping=subtyping, 
            embed_dim=embed_dim
        )
        
        # Agregar sampler topológico
        if diversity_method == 'fps':
            self.topological_sampler = TopologicalDiversitySampler(alpha=diversity_alpha)
        elif diversity_method == 'dpp':
            self.topological_sampler = DPPSampler(lambda_reg=0.01)
        elif diversity_method == 'maxmin':
            self.topological_sampler = MaxMinSampler()
        else:
            raise ValueError(f"Unknown diversity_method: {diversity_method}")
        
        self.diversity_method = diversity_method
    
    def inst_eval(self, A, h, classifier):
        """
        Sobrescribe inst_eval para CLAM_MB con diversidad topológica
        """
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        
        A_flat = A.view(-1)
        
        # Seleccionar con diversidad topológica
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
        """
        Sobrescribe inst_eval_out para CLAM_MB con diversidad topológica
        """
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