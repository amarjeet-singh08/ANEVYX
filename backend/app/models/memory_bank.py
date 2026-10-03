import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalityMemoryBank(nn.Module):
    """
    Stores embeddings of normal scans during training to use as a nearest-neighbor 
    reference during inference.
    """
    def __init__(self, embed_dim=512, capacity=10000):
        super().__init__()
        self.capacity = capacity
        self.embed_dim = embed_dim
        
        # Register as a buffer so it's saved with the model state_dict but not updated by backprop
        self.register_buffer('bank', torch.empty(0, embed_dim))
        
    def add_features(self, features: torch.Tensor):
        """
        Add batch of features to memory bank. 
        features: [B, embed_dim]
        """
        B = features.shape[0]
        # Normalize features for cosine distance
        features = F.normalize(features, p=2, dim=1)
        
        if self.bank.shape[0] == 0:
            self.bank = features.detach()
        else:
            self.bank = torch.cat([self.bank, features.detach()], dim=0)
            
        # Enforce capacity (keep latest or random; here we just keep latest)
        if self.bank.shape[0] > self.capacity:
            self.bank = self.bank[-self.capacity:]
            
    @property
    def is_fitted(self) -> bool:
        return self.bank.shape[0] > 0

    def compute_distance(self, query: torch.Tensor, k: int = 5) -> torch.Tensor:
        """
        Compute distance between query and memory bank.
        query: [B, embed_dim]
        Returns: [B] distance scores
        """
        if not self.is_fitted:
            return None
            
        query = F.normalize(query, p=2, dim=1)
        
        # Ensure bank is also normalized (in case raw features were loaded from an old checkpoint)
        bank_norm = F.normalize(self.bank, p=2, dim=1)
        
        # Cosine similarity matrix [B, N]
        sim_matrix = torch.matmul(query, bank_norm.T)
        
        # Get top-k similarities
        k = min(k, self.bank.shape[0])
        topk_sim, _ = torch.topk(sim_matrix, k, dim=1)
        
        # Distance = 1 - similarity (clamped to 0 to prevent float noise)
        dist = torch.clamp(1.0 - topk_sim.mean(dim=1), min=0.0)
        return dist
