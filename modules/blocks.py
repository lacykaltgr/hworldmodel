from torchrl.modules import (
    SafeModule,
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential
)
from typing import Optional
from tensordict.nn import InteractionType
from torchrl.data.tensor_specs import CompositeSpec, UnboundedContinuousTensorSpec, TensorSpec

class LocScaleDist(SafeProbabilisticTensorDictSequential):
    
    def __init__(self, 
                 in_keys, 
                 out_key, 
                 net,
                 distribution_class,
                 distribution_kwargs,
                 loc_only = False,
                 spec: Optional[TensorSpec] = None, 
                 default_interaction_type=InteractionType.MODE
        ):
        
        if spec is not None:
            param_spec = {"loc": UnboundedContinuousTensorSpec(spec.shape, spec.device)}
            if not loc_only:
                param_spec.update({"scale": UnboundedContinuousTensorSpec(spec.shape, spec.device)})
            params_spec = CompositeSpec(**param_spec)
            out_spec = CompositeSpec(**{
                    out_key: spec
                })
        else:
            params_spec = None
            out_spec = None
        
        is_next = isinstance(out_key, tuple) and tuple[0] == "next"   
        if loc_only:
            inner_keys = ["next_loc"] if is_next else ["loc"]
        else:
            inner_keys = ["next_loc", "next_scale"] if is_next else ["loc", "scale"]
            
            
        
        super(LocScaleDist, self).__init__(
            SafeModule(
                net,
                in_keys=in_keys,
                out_keys=inner_keys,
                spec=params_spec
            ),
            SafeProbabilisticModule(
                in_keys=inner_keys,
                out_keys=out_key,
                default_interaction_type=default_interaction_type,
                distribution_class=distribution_class,
                distribution_kwargs=distribution_kwargs,
                spec=out_spec
            ),
            
        )