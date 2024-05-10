# modules correctly implemented in original code

from torchrl.modules import (
    ObsEncoder,
    ObsDecoder
)

# modules with minor changes

from .rssm import (
    RSSMPriorV3,
    RSSMPosteriorV3,
    RSSMRollout,
)

from .actor import DreamerActorV3