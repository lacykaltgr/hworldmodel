from abc import ABC, abstractmethod

import torch
from torch import nn
import numpy as np

from .utils import (
    roll,
    batch_std,
)

class Constraint(nn.Module):
    @abstractmethod
    def forward(self, x):
        pass
    

class DiversityContraint(Constraint):
    
    """
    Compute the diversity term for a set of images.

    :param div_metric: What metric to use when computing pairwise differences.
    :param div_linkage: How to agglomerate pairwise distances.
    :param div_weight: Weight given to the diversity term in the objective function.
    :param div_mask: Array (height x width) used to mask irrelevant parts of the
                        image before calculating diversity.
    """
    
    def __init__(self, div_metric='correlation', div_linkage='minimum', div_weight=0, div_mask=1):
        self.div_metric = div_metric
        self.div_linkage = div_linkage
        self.div_weight = div_weight
        self.div_mask = div_mask
        
    def forward(self, x):
        """
        x: (B, ...)
        """
        div_term = 0
        if self.div_weight > 0:
            mask = torch.tensor(self.div_mask, dtype=torch.float32, device=x.device)
            # Compute distance matrix
            x_m = (x * mask).view(len(x), -1)  # num_images x num_pixels    # img: x, images: x_m
            if self.div_metric == 'correlation':
                # computations restricted to the mask
                means = (x_m.sum(dim=-1) / mask.sum()).view(len(x_m), 1, 1, 1)
                residuals = ((x - means) * torch.sqrt(mask)).view(len(x), -1)
                ssr = (((x - means) ** 2) * mask).sum(-1).sum(-1).sum(-1)
                distance_matrix = -(torch.mm(residuals, residuals.t()) /
                                    torch.sqrt(torch.ger(ssr, ssr)) + 1e-12)
            elif self.div_metric == 'cosine':
                image_norms = torch.norm(x_m, dim=-1)
                distance_matrix = -(torch.mm(x_m, x_m.t()) /
                                    (torch.ger(image_norms, image_norms) + 1e-12))
            elif self.div_metric == 'euclidean':
                distance_matrix = torch.norm(x_m.unsqueeze(0) -
                                            x_m.unsqueeze(1), dim=-1)
            else:
                raise ValueError('Invalid distance metric {} for the diversity term'.format(self.div_metric))

            # Compute overall distance in this image set
            triu_idx = torch.triu(torch.ones(len(distance_matrix),
                                            len(distance_matrix)), diagonal=1) == 1
            if self.div_linkage == 'minimum':
                distance = distance_matrix[triu_idx].min()
            elif self.div_linkage == 'average':
                distance = distance_matrix[triu_idx].mean()
            else:
                raise ValueError('Invalid linkage for the diversity term: {}'.format(self.div_linkage))

            div_term = self.div_weight * distance
        return div_term
        
