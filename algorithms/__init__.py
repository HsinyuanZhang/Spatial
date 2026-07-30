from .spatial_features import extract_spatial_features
from .som_clustering import SpatialSOM
from .detection import detect_spikes_neo
from .adaptive_range_search import (
    AdaptiveRangeSearch,
    UniformUnsignedQuantizer,
    evaluate_candidate_search,
    fit_integer_centroids,
)
from .adaptive_prototype_search import AdaptivePrototypeSearch
from .adaptive_masked_range_search import AdaptiveMaskedRangeSearch
from .adaptive_weighted_range_search import AdaptiveWeightedRangeSearch
from .causal_temporal_sketch import CausalTemporalSketch
from .spatial_wta import SpatialWTAClassifier
from .online_spatial_adaptation import OnlineSpatialAdapter
from .geometric_prefilters import (
    causal_first_channel,
    index_neighborhood_unit_mask,
    integer_cosine_scores,
    jaccard_filter_mask,
    proxy_max_channel,
)
