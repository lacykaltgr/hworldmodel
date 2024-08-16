# modules correctly implemented in original code

from torchrl.modules import (
    ObsEncoder,
    ObsDecoder,
)

from .depth_decoder import (
    DepthDecoder
)

# modules with minor changes

from .rssm import (
    RSSMPriorV2,
    RSSMPosteriorV2,
    RSSMRollout,
)

from .mb_env import DreamerEnv
from .actor import DreamerActorV2