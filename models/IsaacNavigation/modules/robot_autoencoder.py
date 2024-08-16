import torch
from torch import nn

import torchrl

from . import ObsEncoder, ObsDecoder
from .depth_decoder import DepthDecoder


class RobotEncoder(nn.Module):
    
    def __init__(self, image_type: str = 'depth', obs_name_dim_dict: dict = None):
        super(RobotEncoder, self).__init__()
        
        self.img_encoder = ObsEncoder()
        if image_type == 'depth':
            self.img_decoder = DepthDecoder()
        elif image_type == 'rgb':
            self.img_decoder = ObsDecoder()
        else:
            raise ValueError(f'Invalid image type: {image_type}')
        
        assert obs_name_dim_dict is not None 
        
        
        
        