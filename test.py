import copy
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

from typing import TypeVar

sDataFrame = TypeVar("sDataFrame")  # type: ignore

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typeguard import typechecked
from visions import VisionsTypeset

from ydata_profiling.config import Config, Settings, SparkSettings
from ydata_profiling.expectations_report import ExpectationsReport
from ydata_profiling.model.summarizer import (
    BaseSummarizer,
    PandasProfilingSummarizer,
)
from ydata_profiling.model.typeset import ProfilingTypeSet
from ydata_profiling.serialize_report import SerializeReport
from ydata_profiling.utils.paths import get_config

from report import get_report_structure
from describe import describe as describe_df



@typechecked
class ProfileReport(SerializeReport, ExpectationsReport):
    """Generate a profile report from a Dataset stored as a pandas `DataFrame`.

    Used as is, it will output its content as an HTML report in a Jupyter notebook.
    """

    _description_set = None
    _report = None
    _html = None
    _widgets = None
    _json = None
    config: Settings

    def __init__(
        self,
        df: Optional[Union[pd.DataFrame, sDataFrame]] = None,
        minimal: bool = False,
        tsmode: bool = False,
        sortby: Optional[str] = None,
        sensitive: bool = False,
        explorative: bool = False,
        dark_mode: bool = False,
        orange_mode: bool = False,
        sample: Optional[dict] = None,
        config_file: Union[Path, str] = None,
        lazy: bool = True,
        typeset: Optional[VisionsTypeset] = None,
        summarizer: Optional[BaseSummarizer] = None,
        config: Optional[Settings] = None,
        type_schema: Optional[dict] = None,
        **kwargs,
    ):
        """Generate a ProfileReport based on a pandas or spark.sql DataFrame

        Config processing order (in case of duplicate entries, entries later in the order are retained):
        - config presets (e.g. `config_file`, `minimal` arguments)
        - config groups (e.g. `explorative` and `sensitive` arguments)
        - custom settings (e.g. `config` argument)
        - custom settings **kwargs (e.g. `title`)

        Args:
            df: a pandas or spark.sql DataFrame
            minimal: minimal mode is a default configuration with minimal computation
            ts_mode: activates time-series analysis for all the numerical variables from the dataset. Only available for pd.DataFrame
            sort_by: ignored if ts_mode=False. Order the dataset by a provided column.
            sensitive: hides the values for categorical and text variables for report privacy
            config_file: a config file (.yml), mutually exclusive with `minimal`
            lazy: compute when needed
            sample: optional dict(name="Sample title", caption="Caption", data=pd.DataFrame())
            typeset: optional user typeset to use for type inference
            summarizer: optional user summarizer to generate custom summary output
            type_schema: optional dict containing pairs of `column name`: `type`
            **kwargs: other arguments, for valid arguments, check the default configuration file.
        """
        self.__validate_inputs(df, minimal, tsmode, config_file, lazy)

        if config_file or minimal:
            if not config_file:
                config_file = get_config("config_minimal.yaml")

            report_config = Settings().from_file(config_file)
        elif config is not None:
            report_config = config
        else:
            if isinstance(df, pd.DataFrame):
                report_config = Settings()
            else:
                report_config = SparkSettings()

        groups = [
            (explorative, "explorative"),
            (sensitive, "sensitive"),
            (dark_mode, "dark_mode"),
            (orange_mode, "orange_mode"),
        ]

        if any(condition for condition, _ in groups):
            cfg = Settings()
            for condition, key in groups:
                if condition:
                    cfg = cfg.update(Config.get_arg_groups(key))
            report_config = cfg.update(report_config.dict(exclude_defaults=True))

        if len(kwargs) > 0:
            shorthands, kwargs = Config.shorthands(kwargs)
            report_config = (
                Settings()
                .update(shorthands)
                .update(report_config.dict(exclude_defaults=True))
            )

        if kwargs:
            report_config = report_config.update(kwargs)

        report_config.vars.timeseries.active = tsmode
        if tsmode and sortby:
            report_config.vars.timeseries.sortby = sortby

        self.df = self.__initialize_dataframe(df, report_config)
        self.config = report_config
        self._df_hash = None
        self._sample = sample
        self._type_schema = type_schema
        self._typeset = typeset
        self._summarizer = summarizer

        if not lazy:
            # Trigger building the report structure
            _ = self.report

    @staticmethod
    def __validate_inputs(
        df: Optional[Union[pd.DataFrame, sDataFrame]],
        minimal: bool,
        tsmode: bool,
        config_file: Optional[Union[Path, str]],
        lazy: bool,
    ) -> None:

        # Lazy profile cannot be set if no DataFrame is provided
        if df is None and not lazy:
            raise ValueError("Can init a not-lazy ProfileReport with no DataFrame")

        if config_file is not None and minimal:
            raise ValueError(
                "Arguments `config_file` and `minimal` are mutually exclusive."
            )

        # Spark Dataframe validations
        if isinstance(df, pd.DataFrame):
            if df is not None and df.empty:
                raise ValueError(
                    "DataFrame is empty. Please" "provide a non-empty DataFrame."
                )
        else:
            if tsmode:
                raise NotImplementedError(
                    "Time-Series dataset analysis is not yet supported for Spark DataFrames"
                )

            if (
                df is not None and df.rdd.isEmpty()
            ):  # df.isEmpty is only support by 3.3.0 pyspark version
                raise ValueError(
                    "DataFrame is empty. Please" "provide a non-empty DataFrame."
                )

    @staticmethod
    def __initialize_dataframe(
        df: Optional[Union[pd.DataFrame, sDataFrame]], report_config: Settings
    ) -> Optional[Union[pd.DataFrame, sDataFrame]]:
        if (
            df is not None
            and isinstance(df, pd.DataFrame)
            and report_config.vars.timeseries.active
            and report_config.vars.timeseries.sortby
        ):
            return df.sort_values(by=report_config.vars.timeseries.sortby).reset_index(
                drop=True
            )
        else:
            return df

    def invalidate_cache(self, subset: Optional[str] = None) -> None:
        """Invalidate report cache. Useful after changing setting.

        Args:
            subset:
            - "rendering" to invalidate the html, json and widget report rendering
            - "report" to remove the caching of the report structure
            - None (default) to invalidate all caches

        Returns:
            None
        """
        if subset is not None and subset not in ["rendering", "report"]:
            raise ValueError(
                "'subset' parameter should be None, 'rendering' or 'report'"
            )

        if subset is None or subset in ["rendering", "report"]:
            self._widgets = None
            self._json = None
            self._html = None

        if subset is None or subset == "report":
            self._report = None

        if subset is None:
            self._description_set = None

    @property
    def typeset(self) -> Optional[VisionsTypeset]:
        if self._typeset is None:
            self._typeset = ProfilingTypeSet(self.config, self._type_schema)
        return self._typeset

    @property
    def summarizer(self) -> BaseSummarizer:
        if self._summarizer is None:
            self._summarizer = PandasProfilingSummarizer(self.typeset)
        return self._summarizer

    @property
    def description_set(self) -> Dict[str, Any]:
        if self._description_set is None:
            self._description_set = describe_df(
                self.config,
                self.df,
                self.summarizer,
                self.typeset,
                self._sample,
            )
        return self._description_set
    
    def run(self):
        df = self.description_set
        return df

    def get_config(self):
        return self.config
    
    def get_typeset(self):
        return self._typeset

    def get_summarizer(self):
        return self._summarizer

if __name__ == "__main__":
    import os
    data_name = 'melb_data.csv'
    DATA_PATH = os.path.join(os.getcwd(), "data","input",data_name)
    df = pd.read_csv(DATA_PATH)
    profile = ProfileReport(df, title="Pandas Profiling Report")
    value = profile.run()