# modules correctly implemented in original code

from torchrl.modules import (
    ObsEncoder
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

from .actor import DreamerActorV2