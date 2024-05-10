# modules correctly implemented in original code

from torchrl.modules import (
    ObsEncoder,
    ObsDecoder,
    #RSSMRollout
)

# modules with minor changes

from .model_based import (
    RSSMPrior,
    RSSMPosterior,
    RSSMRollout,
)

from .actor import DreamerActor