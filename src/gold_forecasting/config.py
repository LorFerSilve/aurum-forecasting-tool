"""Typed loading and central validation for versioned project configuration.

The YAML files deliberately contain the research choices, while this module
enforces the invariants that must remain true across those files.  Keeping the
provider fields generic makes a future broker adapter possible without
weakening the XAU/USD research contract.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from math import isclose
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, TypeVar, cast

import yaml
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from yaml.constructor import ConstructorError


class ConfigError(ValueError):
    """Raised when a YAML configuration cannot be loaded or validated."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects silently overwritten duplicate keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class ConfigModel(BaseModel):
    """Strict base class shared by all configuration schemas."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class RunConfig(ConfigModel):
    seed: int = Field(ge=0, le=4_294_967_295)
    output_root: Path
    data_version: str = Field(min_length=1)


class RootConfig(ConfigModel):
    schema_version: Literal[1]
    protocol_version: str = Field(pattern=r"^v\d+\.\d+$")
    instrument_config: Path
    features_config: Path
    labels_config: Path
    costs_config: Path
    splits_config: Path
    model_spec_config: Path = Field(alias="model_config")
    backtest_config: Path
    run: RunConfig

    @field_validator(
        "instrument_config",
        "features_config",
        "labels_config",
        "costs_config",
        "splits_config",
        "model_spec_config",
        "backtest_config",
    )
    @classmethod
    def component_reference_must_be_yaml(cls, value: Path) -> Path:
        if value.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("component configuration references must be YAML files")
        return value


class InstrumentIdentity(ConfigModel):
    id: Literal["XAU_USD"]
    display_name: str = Field(min_length=1)
    base_asset: Literal["XAU"]
    quote_currency: Literal["USD"]
    price_unit: Literal["USD_per_troy_ounce"]
    market_type: str = Field(min_length=1)


class ProviderConfig(ConfigModel):
    """Provider metadata; identifiers stay open so adapters are replaceable."""

    id: str = Field(min_length=1)
    adapter: str = Field(min_length=1)
    source_symbol: str = Field(min_length=1)
    source_url: AnyHttpUrl
    source_timezone_offset: str = Field(pattern=r"^[+-](?:0\d|1[0-4]):[0-5]\d$")
    source_observes_dst: bool
    timestamp_semantics: Literal["candle_open", "candle_close"]
    price_side: Literal["bid", "ask", "mid"]
    volume_reliable: bool
    market_hours_policy: str = Field(min_length=1)


_TIMEFRAME_PATTERN = re.compile(r"^(?P<count>[1-9]\d*)(?P<unit>min|h|d|wk|mo)$")


def _validate_timeframe(value: str) -> str:
    if not _TIMEFRAME_PATTERN.fullmatch(value):
        raise ValueError(
            "timeframe must use an unambiguous code such as '1min', '3min', '15min', or '1mo'"
        )
    return value


def _fixed_timeframe_minutes(value: str) -> int | None:
    match = _TIMEFRAME_PATTERN.fullmatch(value)
    if match is None:
        return None
    count = int(match.group("count"))
    factor = {"min": 1, "h": 60, "d": 1_440, "wk": 10_080}.get(match.group("unit"))
    return None if factor is None else count * factor


class InstrumentDataConfig(ConfigModel):
    raw_timeframe: str
    derived_timeframes: list[str] = Field(min_length=1)
    normalized_timezone: Literal["UTC"]
    require_complete_candles: Literal[True]
    interpolate_gaps: Literal[False]

    _required_mvp_timeframes: ClassVar[frozenset[str]] = frozenset({"3min", "15min"})

    @field_validator("raw_timeframe")
    @classmethod
    def raw_timeframe_is_valid(cls, value: str) -> str:
        value = _validate_timeframe(value)
        if value != "1min":
            raise ValueError("the XAU/USD MVP raw timeframe must be '1min'")
        return value

    @field_validator("derived_timeframes")
    @classmethod
    def derived_timeframes_are_valid(cls, values: list[str]) -> list[str]:
        checked = [_validate_timeframe(value) for value in values]
        if len(checked) != len(set(checked)):
            raise ValueError("derived timeframes must be unique")
        missing = cls._required_mvp_timeframes.difference(checked)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"derived timeframes must include {missing_text}")
        if "1min" in checked:
            raise ValueError("the raw timeframe must not also be a derived timeframe")
        for timeframe in checked:
            if timeframe not in {"3min", "5min", "15min", "30min", "1h", "3h", "1d", "1mo"}:
                raise ValueError(f"unsupported derived timeframe: {timeframe!r}")
            minutes = _fixed_timeframe_minutes(timeframe)
            if minutes is not None and minutes % 1 != 0:
                raise ValueError("derived timeframes must align to the raw timeframe")
        return checked


class InstrumentConfig(ConfigModel):
    schema_version: Literal[1]
    instrument: InstrumentIdentity
    provider: ProviderConfig
    data: InstrumentDataConfig


def _positive_unique_sorted(values: list[int], *, field_name: str) -> list[int]:
    if any(value <= 0 for value in values):
        raise ValueError(f"{field_name} must contain only positive integers")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    if values != sorted(values):
        raise ValueError(f"{field_name} must be sorted in ascending order")
    return values


class FeaturesConfig(ConfigModel):
    """Frozen, auditable specification for the price-only MVP features."""

    schema_version: Literal[1]
    input_timeframe: str
    return_lags: list[int] = Field(min_length=1)
    momentum_windows: list[int] = Field(min_length=2, max_length=2)
    realized_volatility_window: int = Field(gt=1)
    moving_average_windows: list[int] = Field(min_length=1, max_length=2)
    zero_range_close_position: float = Field(ge=0, le=1)
    tick_count_enabled: bool

    @field_validator("input_timeframe")
    @classmethod
    def input_timeframe_is_valid(cls, value: str) -> str:
        return _validate_timeframe(value)

    @field_validator("return_lags", "momentum_windows", "moving_average_windows")
    @classmethod
    def windows_are_positive_unique_and_sorted(
        cls,
        values: list[int],
        info: Any,
    ) -> list[int]:
        return _positive_unique_sorted(values, field_name=info.field_name)

    @property
    def maximum_prior_candles(self) -> int:
        """Number of prior candles required by the longest configured formula."""

        return max(
            max(self.return_lags) + 1,
            max(self.momentum_windows),
            self.realized_volatility_window,
            max(self.moving_average_windows),
        )


class PredictionConfig(ConfigModel):
    cadence: str
    alignment_timezone: Literal["UTC"]
    feature_cutoff_rule: Literal["closed_candles_only"]
    decision_latency_minutes: int = Field(ge=1)
    entry_rule: Literal["next_1min_open_after_latency"]
    primary_horizon_minutes: Literal[15]
    exit_rule: Literal["first_1min_open_at_horizon"]

    @field_validator("cadence")
    @classmethod
    def cadence_is_valid(cls, value: str) -> str:
        return _validate_timeframe(value)


class ReturnConfig(ConfigModel):
    type: Literal["log_return"]
    unit: Literal["basis_points"]
    price_side: Literal["bid", "ask", "mid"]
    formula: str = Field(min_length=1)

    @model_validator(mode="after")
    def formula_matches_price_side(self) -> Self:
        expected = f"10000 * ln(exit_{self.price_side}_open / entry_{self.price_side}_open)"
        if self.formula != expected:
            raise ValueError(f"return formula must be {expected!r}")
        return self


class ClassesConfig(ConfigModel):
    names: list[Literal["down", "neutral", "up"]] = Field(min_length=3, max_length=3)
    estimated_cost_buffer_bps: float = Field(ge=0)
    noise_buffer_bps: float = Field(ge=0)
    total_threshold_bps: float = Field(gt=0)
    up_rule: str = Field(min_length=1)
    down_rule: str = Field(min_length=1)
    neutral_rule: str = Field(min_length=1)

    @model_validator(mode="after")
    def buffer_and_rules_are_consistent(self) -> Self:
        if self.names != ["down", "neutral", "up"]:
            raise ValueError("class names must be ordered as down, neutral, up")
        expected_total = self.estimated_cost_buffer_bps + self.noise_buffer_bps
        if not isclose(self.total_threshold_bps, expected_total, abs_tol=1e-9):
            raise ValueError(
                "total_threshold_bps must equal estimated_cost_buffer_bps + noise_buffer_bps"
            )
        threshold = str(self.total_threshold_bps)
        expected_rules = {
            "up_rule": f"gross_return_bps > {threshold}",
            "down_rule": f"gross_return_bps < -{threshold}",
            "neutral_rule": (f"-{threshold} <= gross_return_bps <= {threshold}"),
        }
        for field_name, expected in expected_rules.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} must be {expected!r}")
        return self


class InvalidSampleConfig(ConfigModel):
    reject_missing_entry: Literal[True]
    reject_missing_exit: Literal[True]
    reject_incomplete_horizon: Literal[True]
    reject_disallowed_gap_crossing: Literal[True]


class LabelsConfig(ConfigModel):
    schema_version: Literal[1]
    prediction: PredictionConfig
    return_config: ReturnConfig = Field(alias="return")
    classes: ClassesConfig
    invalid_sample: InvalidSampleConfig


class CostModel(ConfigModel):
    name: str = Field(min_length=1)
    unit: Literal["basis_points"]
    source_price_side: Literal["bid", "ask", "mid"]
    estimated_spread_round_trip_bps: float = Field(ge=0)
    slippage_per_side_bps: float = Field(ge=0)
    commission_round_trip_bps: float = Field(ge=0)
    financing_round_trip_bps: float = Field(ge=0)
    total_round_trip_bps: float = Field(ge=0)
    normalized_notional: float = Field(gt=0)
    leverage: float = Field(gt=0)

    @model_validator(mode="after")
    def total_is_sum_of_components(self) -> Self:
        expected = (
            self.estimated_spread_round_trip_bps
            + 2 * self.slippage_per_side_bps
            + self.commission_round_trip_bps
            + self.financing_round_trip_bps
        )
        if not isclose(self.total_round_trip_bps, expected, abs_tol=1e-9):
            raise ValueError(
                "cost_model.total_round_trip_bps must equal spread + two-sided "
                "slippage + commission + financing"
            )
        return self


class StressModel(ConfigModel):
    spread_multiplier: float = Field(gt=0)
    slippage_per_side_bps: float = Field(ge=0)
    total_round_trip_bps: float = Field(ge=0)


class ReportingConfig(ConfigModel):
    include_gross_return: Literal[True]
    include_net_return: Literal[True]
    include_cost_breakdown: Literal[True]


class CostsConfig(ConfigModel):
    schema_version: Literal[1]
    cost_model: CostModel
    stress_model: StressModel
    reporting: ReportingConfig

    @model_validator(mode="after")
    def stress_total_is_consistent(self) -> Self:
        base = self.cost_model
        stress = self.stress_model
        expected = (
            base.estimated_spread_round_trip_bps * stress.spread_multiplier
            + 2 * stress.slippage_per_side_bps
            + base.commission_round_trip_bps
            + base.financing_round_trip_bps
        )
        if not isclose(stress.total_round_trip_bps, expected, abs_tol=1e-9):
            raise ValueError(
                "stress_model.total_round_trip_bps must equal stressed spread + "
                "two-sided stress slippage + commission + financing"
            )
        return self


class TimeRange(ConfigModel):
    start: datetime
    end: datetime

    @field_validator("start", "end")
    @classmethod
    def boundary_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("split boundaries must be timezone-aware UTC timestamps")
        if value.utcoffset() != timedelta(0):
            raise ValueError("split boundaries must be UTC timestamps")
        return value

    @model_validator(mode="after")
    def start_precedes_end(self) -> Self:
        if self.start >= self.end:
            raise ValueError("split start must be before split end")
        return self


class SplitRanges(ConfigModel):
    train: TimeRange
    validation: TimeRange
    test: TimeRange

    @model_validator(mode="after")
    def ranges_are_chronological_and_disjoint(self) -> Self:
        if self.train.end > self.validation.start:
            raise ValueError("train must end no later than validation starts")
        if self.validation.end > self.test.start:
            raise ValueError("validation must end no later than test starts")
        return self


class PurgingConfig(ConfigModel):
    gap_minutes: int = Field(ge=0)
    reason: str = Field(min_length=1)


class DevelopmentGuardConfig(ConfigModel):
    reject_at_or_after: datetime
    allow_holdout_override: Literal[False]

    @field_validator("reject_at_or_after")
    @classmethod
    def guard_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("development holdout guard must be timezone-aware")
        if value.utcoffset() != timedelta(0):
            raise ValueError("development holdout guard must be in UTC")
        return value


class PreprocessingConfig(ConfigModel):
    fit_on: Literal["train"]
    random_shuffle: Literal[False]


class SplitsConfig(ConfigModel):
    schema_version: Literal[1]
    timezone: Literal["UTC"]
    boundary_semantics: Literal["start_inclusive_end_exclusive"]
    splits: SplitRanges
    purging: PurgingConfig
    development_guard: DevelopmentGuardConfig
    preprocessing: PreprocessingConfig

    @model_validator(mode="after")
    def development_ranges_precede_holdout(self) -> Self:
        guard = self.development_guard.reject_at_or_after
        if self.splits.test.end > guard:
            raise ValueError("test must end no later than development_guard.reject_at_or_after")
        return self


class LogisticModelConfig(ConfigModel):
    c_values: list[float] = Field(min_length=1)
    class_weight_options: list[Literal["none", "balanced"]] = Field(min_length=1)
    max_iter: int = Field(gt=0)
    selection_primary: Literal["macro_f1"]
    selection_tiebreak: Literal["log_loss"]
    final_fit: Literal["train_only"]

    @field_validator("c_values")
    @classmethod
    def c_values_are_positive_unique(cls, values: list[float]) -> list[float]:
        if any(not value > 0 for value in values):
            raise ValueError("logistic C values must be positive")
        if len(values) != len(set(values)):
            raise ValueError("logistic C values must be unique")
        return values

    @field_validator("class_weight_options")
    @classmethod
    def class_weight_options_are_unique(
        cls,
        values: list[Literal["none", "balanced"]],
    ) -> list[Literal["none", "balanced"]]:
        if len(values) != len(set(values)):
            raise ValueError("class-weight options must be unique")
        return values


class EvaluationConfig(ConfigModel):
    reliability_buckets: int = Field(ge=2, le=100)
    probability_status: Literal["preliminary"]


class MVPModelConfig(ConfigModel):
    schema_version: Literal[1]
    release_version: str = Field(pattern=r"^v\d+\.\d+\.\d+$")
    class_order: list[Literal["down", "neutral", "up"]] = Field(
        min_length=3, max_length=3
    )
    logistic: LogisticModelConfig
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def class_order_is_frozen(self) -> Self:
        if self.class_order != ["down", "neutral", "up"]:
            raise ValueError("model class_order must be down, neutral, up")
        return self


class BacktestPolicyConfig(ConfigModel):
    confidence_threshold: float = Field(ge=0, le=1)
    position_policy: Literal["single_non_overlapping"]
    normalized_notional: float = Field(gt=0)
    tune_threshold_on: Literal["none"]


class BacktestConfig(ConfigModel):
    schema_version: Literal[1]
    backtest: BacktestPolicyConfig


class ProjectConfig(ConfigModel):
    """Fully loaded configuration plus resolved source paths."""

    config_path: Path
    component_paths: dict[str, Path]
    root: RootConfig
    instrument: InstrumentConfig
    features: FeaturesConfig
    labels: LabelsConfig
    costs: CostsConfig
    splits: SplitsConfig
    model: MVPModelConfig
    backtest: BacktestConfig

    @model_validator(mode="after")
    def components_share_one_research_contract(self) -> Self:
        if self.instrument.instrument.id != "XAU_USD":
            raise ValueError("the project instrument must be XAU_USD")

        data = self.instrument.data
        features = self.features
        prediction = self.labels.prediction
        if prediction.cadence not in data.derived_timeframes:
            raise ValueError("prediction cadence must be one of the derived timeframes")
        if features.input_timeframe != prediction.cadence:
            raise ValueError("feature input timeframe must equal the prediction cadence")
        if features.input_timeframe not in data.derived_timeframes:
            raise ValueError("feature input timeframe must be one of the derived timeframes")
        if features.tick_count_enabled and not self.instrument.provider.volume_reliable:
            raise ValueError("tick-count features require a provider with reliable volume")
        if self.model.class_order != self.labels.classes.names:
            raise ValueError("model and label class order must be identical")
        if not isclose(
            self.backtest.backtest.normalized_notional,
            self.costs.cost_model.normalized_notional,
            abs_tol=1e-12,
        ):
            raise ValueError("backtest notional must equal the cost-model notional")
        horizon_timeframe = f"{prediction.primary_horizon_minutes}min"
        if horizon_timeframe not in data.derived_timeframes:
            raise ValueError("the primary 15-minute horizon must have a matching derived timeframe")

        provider_side = self.instrument.provider.price_side
        return_side = self.labels.return_config.price_side
        cost_side = self.costs.cost_model.source_price_side
        if len({provider_side, return_side, cost_side}) != 1:
            raise ValueError("provider, return, and cost model must use the same price side")

        label_cost = self.labels.classes.estimated_cost_buffer_bps
        configured_cost = self.costs.cost_model.total_round_trip_bps
        if not isclose(label_cost, configured_cost, abs_tol=1e-9):
            raise ValueError(
                "label estimated_cost_buffer_bps must equal the base "
                "cost_model.total_round_trip_bps"
            )

        minimum_gap = prediction.decision_latency_minutes + prediction.primary_horizon_minutes
        if self.splits.purging.gap_minutes < minimum_gap:
            raise ValueError(
                f"purging gap must be at least {minimum_gap} minutes "
                "(decision latency + label horizon)"
            )
        return self


ModelT = TypeVar("ModelT", bound=BaseModel)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ConfigError(f"Cannot read configuration file {path}: {exc}") from exc
    try:
        loaded = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in configuration file {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"Configuration file {path} must contain a YAML mapping")
    if not all(isinstance(key, str) for key in loaded):
        raise ConfigError(f"Configuration file {path} must use string keys")
    return loaded


def _load_model(path: Path, model_type: type[ModelT]) -> ModelT:
    payload = _load_yaml_mapping(path)
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {path}:\n{exc}") from exc


def _resolve_component_path(root_directory: Path, reference: Path) -> Path:
    candidate = reference if reference.is_absolute() else root_directory / reference
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise ConfigError(
            f"Referenced component configuration does not exist: {candidate}"
        ) from exc


def load_project_config(config_path: str | Path) -> ProjectConfig:
    """Load the root YAML and all referenced component YAML files.

    Relative component paths are interpreted relative to the root config, not
    relative to the process working directory.  Every file is validated before
    the cross-file research invariants are checked.
    """

    requested_path = Path(config_path).expanduser()
    try:
        resolved_root = requested_path.resolve(strict=True)
    except OSError as exc:
        raise ConfigError(f"Root configuration does not exist: {requested_path}") from exc
    if not resolved_root.is_file():
        raise ConfigError(f"Root configuration is not a file: {resolved_root}")

    root = _load_model(resolved_root, RootConfig)
    references: dict[str, tuple[Path, type[BaseModel]]] = {
        "instrument": (root.instrument_config, InstrumentConfig),
        "features": (root.features_config, FeaturesConfig),
        "labels": (root.labels_config, LabelsConfig),
        "costs": (root.costs_config, CostsConfig),
        "splits": (root.splits_config, SplitsConfig),
        "model": (root.model_spec_config, MVPModelConfig),
        "backtest": (root.backtest_config, BacktestConfig),
    }
    component_paths: dict[str, Path] = {}
    components: dict[str, BaseModel] = {}
    for name, (reference, model_type) in references.items():
        component_path = _resolve_component_path(resolved_root.parent, reference)
        if not component_path.is_file():
            raise ConfigError(f"Referenced {name} configuration is not a file: {component_path}")
        component_paths[name] = component_path
        components[name] = _load_model(component_path, model_type)

    try:
        return ProjectConfig(
            config_path=resolved_root,
            component_paths=component_paths,
            root=root,
            instrument=cast(InstrumentConfig, components["instrument"]),
            features=cast(FeaturesConfig, components["features"]),
            labels=cast(LabelsConfig, components["labels"]),
            costs=cast(CostsConfig, components["costs"]),
            splits=cast(SplitsConfig, components["splits"]),
            model=cast(MVPModelConfig, components["model"]),
            backtest=cast(BacktestConfig, components["backtest"]),
        )
    except ValidationError as exc:
        raise ConfigError(
            f"Configuration files do not form a valid project contract:\n{exc}"
        ) from exc


__all__ = [
    "BacktestConfig",
    "ConfigError",
    "CostsConfig",
    "FeaturesConfig",
    "InstrumentConfig",
    "LabelsConfig",
    "MVPModelConfig",
    "ProjectConfig",
    "RootConfig",
    "SplitsConfig",
    "load_project_config",
]
