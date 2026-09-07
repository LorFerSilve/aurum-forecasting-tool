"""Leakage-safe feature construction for research pipelines."""

from gold_forecasting.features.mvp import (
    FEATURE_METADATA_COLUMNS,
    FeatureBuildError,
    FeatureBuildResult,
    FeatureCatalog,
    FeatureDefinition,
    build_feature_catalog,
    build_mvp_features,
)
from gold_forecasting.features.phase7 import (
    Phase7FeatureBuildError,
    Phase7FeatureBuildResult,
    Phase7FeatureCatalog,
    Phase7FeatureConfig,
    Phase7FeatureDefinition,
    build_phase7_feature_row,
    build_phase7_features,
    load_phase7_feature_config,
)

__all__ = [
    "FEATURE_METADATA_COLUMNS",
    "FeatureBuildError",
    "FeatureBuildResult",
    "FeatureCatalog",
    "FeatureDefinition",
    "Phase7FeatureBuildError",
    "Phase7FeatureBuildResult",
    "Phase7FeatureCatalog",
    "Phase7FeatureConfig",
    "Phase7FeatureDefinition",
    "build_feature_catalog",
    "build_mvp_features",
    "build_phase7_feature_row",
    "build_phase7_features",
    "load_phase7_feature_config",
]
