import ncps
from ncps.torch import CfCCell, LTCCell, WiredCfCCell
from typing import Optional
from torch.nn import Module

import torch
from torch import Tensor



class LiquidCell(Module):

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int,
        cell_type='CfC', # LTC or CfC
        wiring='fully-connected', # 'fully-connected', 'random' or ncps.wirings.Wiring
            
        cfc_mode="default", # "default", "pure", "no_gate"
        cfc_activation="lecun_tanh", # # silu, relu, gelu, tanh, lecun_tanh
        cfc_backbone_units=None, # 128
        cfc_backbone_layers=None, # 1
        cfc_backbone_dropout=None, # 0.0
        
        ltc_ode_unfolds=6, # 1
        ltc_implicit_param_constraints=True, # else call apply_weight_constraints() to enforce constraints
    ) -> None:
        
        super(LiquidCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if wiring == 'fully-connected':
            self.wiring = ncps.wirings.FullyConnected(hidden_size)
        elif wiring == 'random':
            self.wiring = ncps.wirings.Random(hidden_size)
        elif isinstance(wiring, ncps.wirings.Wiring):
            self.wiring = wiring
        else:
            raise ValueError(f"Unknown wiring: {wiring}")
        
        if cell_type == "CfC":
            
            if wiring != 'fully-connected':
                if cfc_backbone_units is not None:
                    raise ValueError(f"Cannot use backbone_units in wired mode")
                if cfc_backbone_layers is not None:
                    raise ValueError(f"Cannot use backbone_layers in wired mode")
                if cfc_backbone_dropout is not None:
                    raise ValueError(f"Cannot use backbone_dropout in wired mode")
                
                self.rnn_cell = WiredCfCCell(
                    input_size,
                    self.wiring,
                    cfc_mode,
                )
            else:
                self.wired_false = True
                backbone_units = 128 if backbone_units is None else backbone_units
                backbone_layers = 1 if backbone_layers is None else backbone_layers
                backbone_dropout = 0.0 if backbone_dropout is None else backbone_dropout

                self.rnn_cell = CfCCell(
                    input_size,
                    self.wiring,
                    cfc_mode,
                    cfc_activation,
                    backbone_units,
                    backbone_layers,
                    backbone_dropout,
                )
                
        elif cell_type == "LTC":
            
            self.rnn_cell = LTCCell(
                wiring=self.wiring,
                in_features=input_size,
                input_mapping='none', # 'affine', 'linear', 'none
                output_mapping='none', # 'affine', 'linear', 'none'
                ode_unfolds=ltc_ode_unfolds,
                epsilon=1e-8,
                implicit_param_constraints=ltc_implicit_param_constraints,
            )

    def forward(self, input: Tensor, hx: Optional[Tensor] = None, ts: float = 1.0) -> Tensor:
        if input.dim() not in (1, 2):
            raise ValueError(f"LiquidCell: Expected input to be 1D or 2D, got {input.dim()}D instead")
        if hx is not None and hx.dim() not in (1, 2):
            raise ValueError(f"LiquidCell: Expected hidden to be 1D or 2D, got {hx.dim()}D instead")
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        _, ret = self.rnn_cell(input, hx, ts)

        if not is_batched:
            ret = ret.squeeze(0)

        return ret