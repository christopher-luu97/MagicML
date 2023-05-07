# from ydata_profiling/visualisation/utils.py
# from ydata_profiling/visualisation/plot.py
# from ydata_profiling/model/pairwise.py
import base64
import uuid
import io
import copy
import contextlib
import pickle
import warnings
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
from urllib.parse import quote
from datetime import datetime, timedelta
from typeguard import typechecked
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.cbook
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap, rgb2hex
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.artist import Artist
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import (
    deregister_matplotlib_converters,
    register_matplotlib_converters,
)
from ydata_profiling.config import Settings

def convert_timestamp_to_datetime(timestamp: int) -> datetime:
    if timestamp >= 0:
        return datetime.fromtimestamp(timestamp)
    else:
        return datetime(1970, 1, 1) + timedelta(seconds=int(timestamp))
    
@contextlib.contextmanager
def manage_matplotlib_context() -> Any:
    """Return a context manager for temporarily changing matplotlib unit registries and rcParams."""
    originalRcParams = matplotlib.rcParams.copy()

    # Credits for this style go to the ggplot and seaborn packages.
    #   We copied the style file to remove dependencies on the Seaborn package.
    #   Check it out, it's an awesome library for plotting
    customRcParams = {
        "patch.facecolor": "#348ABD",  # blue
        "patch.antialiased": True,
        "font.size": 10.0,
        "figure.edgecolor": "0.50",
        # Seaborn common parameters
        "figure.facecolor": "white",
        "text.color": ".15",
        "axes.labelcolor": ".15",
        "legend.numpoints": 1,
        "legend.scatterpoints": 1,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": ".15",
        "ytick.color": ".15",
        "axes.axisbelow": True,
        "image.cmap": "Greys",
        "font.family": ["sans-serif"],
        "font.sans-serif": [
            "Arial",
            "Liberation Sans",
            "Bitstream Vera Sans",
            "sans-serif",
        ],
        "grid.linestyle": "-",
        "lines.solid_capstyle": "round",
        # Seaborn darkgrid parameters
        # .15 = dark_gray
        # .8 = light_gray
        "axes.grid": True,
        "axes.facecolor": "#EAEAF2",
        "axes.edgecolor": "white",
        "axes.linewidth": 0,
        "grid.color": "white",
        # Seaborn notebook context
        "figure.figsize": [8.0, 5.5],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "grid.linewidth": 1,
        "lines.linewidth": 1.75,
        "patch.linewidth": 0.3,
        "lines.markersize": 7,
        "lines.markeredgewidth": 0,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.pad": 7,
        "ytick.major.pad": 7,
        "backend": "agg",
    }

    try:
        register_matplotlib_converters()
        matplotlib.rcParams.update(customRcParams)
        sns.set_style(style="white")
        yield
    finally:
        deregister_matplotlib_converters()  # revert to original unit registries
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
            matplotlib.rcParams.update(originalRcParams)  # revert to original rcParams
    
def hex_to_rgb(hex: str) -> Tuple[float, ...]:
    """Format a hex value (#FFFFFF) as normalized RGB (1.0, 1.0, 1.0).

    Args:
        hex: The hex value.

    Returns:
        The RGB representation of that hex color value.
    """
    hex = hex.lstrip("#")
    hlen = len(hex)
    return tuple(
        int(hex[i : i + hlen // 3], 16) / 255 for i in range(0, hlen, hlen // 3)
    )


def base64_image(image: bytes, mime_type: str) -> str:
    """Encode the image for an URL using base64

    Args:
        image: the image
        mime_type: the mime type

    Returns:
        A string starting with "data:{mime_type};base64,"
    """
    base64_data = base64.b64encode(image)
    image_data = quote(base64_data)
    return f"data:{mime_type};base64,{image_data}"


# We want to encode the image so we can easily transfer it across devices
def base64_encode_plt(image):
    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()
    


def plot_360_n0sc0pe(
    config: Settings,
    image_format: Optional[str] = None,
    bbox_extra_artists: Optional[List[Artist]] = None,
    bbox_inches: Optional[str] = None,
    fig_handle: Optional[object] = None,
) -> str:
    """Quickscope the plot to a base64 encoded string.

    Args:
        config: Settings
        image_format: png or svg, overrides config.

    Returns:
        A base64 encoded version of the plot in the specified image format.
    """

    if image_format is None:
        image_format = config.plot.image_format.value

    mime_types = {"png": "image/png", "svg": "image/svg+xml"}
    if image_format not in mime_types:
        raise ValueError('Can only 360 n0sc0pe "png" or "svg" format.')
    
    if config.html.inline:
        if image_format == "svg":
            image_str = StringIO()

            plt.savefig(
                image_str,
                format=image_format,
                bbox_extra_artists=bbox_extra_artists,
                bbox_inches=bbox_inches,
            )
            plt.close()
            result_string = image_str.getvalue()
        else:
            image_bytes = BytesIO()
            plt.savefig(
                image_bytes,
                dpi=config.plot.dpi,
                format=image_format,
                bbox_extra_artists=bbox_extra_artists,
                bbox_inches=bbox_inches,
            )
            plt.close()
            result_string = base64_image(
                image_bytes.getvalue(), mime_types[image_format]
            )
            result_string = pickle.dumps(fig_handle)
    else:
        if config.html.assets_path is None:
            raise ValueError("config.html.assets_path may not be none")

        file_path = Path(config.html.assets_path)
        suffix = f"{config.html.assets_prefix}/images/{uuid.uuid4().hex}.{image_format}"
        args = {
            "fname": file_path / suffix,
            "format": image_format,
        }

        if image_format == "png":
            args["dpi"] = config.plot.dpi
        plt.savefig(
            bbox_extra_artists=bbox_extra_artists, bbox_inches=bbox_inches, **args
        )
        plt.close()
        result_string = suffix
    return result_string

def format_fn(tick_val: int, tick_pos: Any) -> str:
    return convert_timestamp_to_datetime(tick_val).strftime("%Y-%m-%d %H:%M:%S")


def _plot_histogram(
    config: Settings,
    series: np.ndarray,
    bins: Union[int, np.ndarray],
    figsize: tuple = (6, 4),
    date: bool = False,
    hide_yaxis: bool = False,
) -> plt.Figure:
    """Plot a histogram from the data and return the AxesSubplot object.

    Args:
        config: the Settings object
        series: The data to plot
        bins: number of bins (int for equal size, ndarray for variable size)
        figsize: The size of the figure (width, height) in inches, default (6,4)
        date: is the x-axis of date type

    Returns:
        The histogram plot.
    """
    # we have precomputed the histograms...
    if isinstance(bins, list):
        n_labels = len(config.html.style._labels)
        fig = plt.figure(figsize=figsize)
        plot = fig.add_subplot(111)

        for idx in reversed(list(range(n_labels))):
            diff = np.diff(bins[idx])
            plot.bar(
                bins[idx][:-1] + diff / 2,  # type: ignore
                series[idx],
                diff,
                facecolor=config.html.style.primary_colors[idx],
                alpha=0.6,
            )

            if date:
                plot.xaxis.set_major_formatter(FuncFormatter(format_fn))

            if not config.plot.histogram.x_axis_labels:
                plot.set_xticklabels([])

            if hide_yaxis:
                plot.yaxis.set_visible(False)

        if not config.plot.histogram.x_axis_labels:
            fig.xticklabels([])

        if not hide_yaxis:
            fig.supylabel("Frequency")
    else:
        fig = plt.figure(figsize=figsize)
        plot = fig.add_subplot(111)
        if not hide_yaxis:
            plot.set_ylabel("Frequency")
        else:
            plot.axes.get_yaxis().set_visible(False)

        diff = np.diff(bins)
        plot.bar(
            bins[:-1] + diff / 2,  # type: ignore
            series,
            diff,
            facecolor=config.html.style.primary_colors[0],
        )

        if date:
            plot.xaxis.set_major_formatter(FuncFormatter(format_fn))

        if not config.plot.histogram.x_axis_labels:
            plot.set_xticklabels([])

    return plot


@manage_matplotlib_context()
def histogram(
    config: Settings,
    series: np.ndarray,
    bins: Union[int, np.ndarray],
    date: bool = False,
) -> str:
    """Plot an histogram of the data.

    Args:
        config: Settings
        series: The data to plot.
        bins: number of bins (int for equal size, ndarray for variable size)
        date: is histogram of date(time)?

    Returns:
      The resulting histogram encoded as a string.

    """
    plot = _plot_histogram(config, series, bins, date=date, figsize=(7, 3))
    plot.xaxis.set_tick_params(rotation=90 if date else 45)
    plot.figure.tight_layout()
    return plot_360_n0sc0pe(config)


@manage_matplotlib_context()
def mini_histogram(
    config: Settings,
    series: np.ndarray,
    bins: Union[int, np.ndarray],
    date: bool = False,
) -> str:
    """Plot a small (mini) histogram of the data.

    Args:
      config: Settings
      series: The data to plot.
      bins: number of bins (int for equal size, ndarray for variable size)

    Returns:
      The resulting mini histogram encoded as a string.
    """
    plot = _plot_histogram(
        config, series, bins, figsize=(3, 2.25), date=date, hide_yaxis=True
    )
    plot.set_facecolor("w")

    for tick in plot.xaxis.get_major_ticks():
        tick.label1.set_fontsize(6 if date else 8)
    plot.xaxis.set_tick_params(rotation=90 if date else 45)
    plot.figure.tight_layout()

    return plot_360_n0sc0pe(config)


def get_cmap_half(
    cmap: Union[Colormap, LinearSegmentedColormap, ListedColormap]
) -> LinearSegmentedColormap:
    """Get the upper half of the color map

    Args:
        cmap: the color map

    Returns:
        A new color map based on the upper half of another color map

    References:
        https://stackoverflow.com/a/24746399/470433
    """
    # Evaluate an existing colormap from 0.5 (midpoint) to 1 (upper end)
    colors = cmap(np.linspace(0.5, 1, cmap.N // 2))

    # Create a new colormap from those colors
    return LinearSegmentedColormap.from_list("cmap_half", colors)


def get_correlation_font_size(n_labels: int) -> Optional[int]:
    """Dynamic label font sizes in correlation plots

    Args:
        n_labels: the number of labels

    Returns:
        A font size or None for the default font size
    """
    if n_labels > 100:
        font_size = 4
    elif n_labels > 80:
        font_size = 5
    elif n_labels > 50:
        font_size = 6
    elif n_labels > 40:
        font_size = 8
    else:
        return None
    return font_size


@manage_matplotlib_context()
def correlation_matrix(config: Settings, data: pd.DataFrame, vmin: int = -1) -> str:
    """Plot image of a matrix correlation.

    Args:
      config: Settings
      data: The matrix correlation to plot.
      vmin: Minimum value of value range.

    Returns:
      The resulting correlation matrix encoded as a string.
    """
    fig_cor, axes_cor = plt.subplots()

    cmap = plt.get_cmap(config.plot.correlation.cmap)
    if vmin == 0:
        cmap = get_cmap_half(cmap)
    cmap = copy.copy(cmap)
    cmap.set_bad(config.plot.correlation.bad)

    labels = data.columns
    matrix_image = axes_cor.imshow(
        data, vmin=vmin, vmax=1, interpolation="nearest", cmap=cmap
    )
    plt.colorbar(matrix_image)

    if data.isnull().values.any():
        legend_elements = [Patch(facecolor=cmap(np.nan), label="invalid\ncoefficient")]

        plt.legend(
            handles=legend_elements,
            loc="upper right",
            handleheight=2.5,
        )

    axes_cor.set_xticks(np.arange(0, data.shape[0], float(data.shape[0]) / len(labels)))
    axes_cor.set_yticks(np.arange(0, data.shape[1], float(data.shape[1]) / len(labels)))

    font_size = get_correlation_font_size(len(labels))
    axes_cor.set_xticklabels(labels, rotation=90, fontsize=font_size)
    axes_cor.set_yticklabels(labels, fontsize=font_size)
    plt.subplots_adjust(bottom=0.2)

    return plot_360_n0sc0pe(config)


@manage_matplotlib_context()
def scatter_complex(config: Settings, series: pd.Series) -> str:
    """Scatter plot (or hexbin plot) from a series of complex values

    Examples:
        >>> complex_series = pd.Series([complex(1, 3), complex(3, 1)])
        >>> scatter_complex(complex_series)

    Args:
        config: Settings
        series: the Series

    Returns:
        A string containing (a reference to) the image
    """
    plt.ylabel("Imaginary")
    plt.xlabel("Real")

    color = config.html.style.primary_colors[0]

    if len(series) > config.plot.scatter_threshold:
        cmap = sns.light_palette(color, as_cmap=True)
        plt.hexbin(series.real, series.imag, cmap=cmap)
    else:
        plt.scatter(series.real, series.imag, color=color)

    return plot_360_n0sc0pe(config)


@manage_matplotlib_context()
def scatter_series(
    config: Settings, series: pd.Series, x_label: str = "Width", y_label: str = "Height"
) -> str:
    """Scatter plot (or hexbin plot) from one series of sequences with length 2

    Examples:
        >>> scatter_series(file_sizes, "Width", "Height")

    Args:
        config: report Settings object
        series: the Series
        x_label: the label on the x-axis
        y_label: the label on the y-axis

    Returns:
        A string containing (a reference to) the image
    """
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    color = config.html.style.primary_colors[0]

    data = zip(*series.tolist())
    if len(series) > config.plot.scatter_threshold:
        cmap = sns.light_palette(color, as_cmap=True)
        plt.hexbin(*data, cmap=cmap)
    else:
        plt.scatter(*data, color=color)
    return plot_360_n0sc0pe(config)


def get_scatter_tasks(
    config: Settings, continuous_variables: list
) -> List[Tuple[Any, Any]]:
    if not config.interactions.continuous:
        return []

    targets = config.interactions.targets
    if len(targets) == 0:
        targets = continuous_variables

    tasks = [(x, y) for y in continuous_variables for x in targets]
    return tasks

def get_scatter_plot(
    config: Settings, df: pd.DataFrame, x: Any, y: Any, continuous_variables: list
) -> str:
    if x in continuous_variables:
        if y == x:
            df_temp = df[[x]].dropna()
        else:
            df_temp = df[[x, y]].dropna()
        return scatter_pairwise(config, df_temp[x], df_temp[y], x, y)
    else:
        return ""

@manage_matplotlib_context()
def scatter_pairwise(
    config: Settings, series1: pd.Series, series2: pd.Series, x_label: str, y_label: str
) -> str:
    """Scatter plot (or hexbin plot) from two series

    Examples:
        >>> widths = pd.Series([800, 1024])
        >>> heights = pd.Series([600, 768])
        >>> scatter_series(widths, heights, "Width", "Height")

    Args:
        config: Settings
        series1: the series corresponding to the x-axis
        series2: the series corresponding to the y-axis
        x_label: the label on the x-axis
        y_label: the label on the y-axis

    Returns:
        A string containing (a reference to) the image
    """
    fig_handle = plt.figure()
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    color = config.html.style.primary_colors[0]

    indices = (series1.notna()) & (series2.notna())
    if len(series1) > config.plot.scatter_threshold:
        cmap = sns.light_palette(color, as_cmap=True)
        plt.hexbin(series1[indices], series2[indices], gridsize=15, cmap=cmap)
    else:
        plt.scatter(series1[indices], series2[indices], color=color)
    fig_obj = pickle.dumps(fig_handle)
    plt.close()
    return  fig_obj


def _plot_stacked_barh(
    data: pd.Series, colors: List, hide_legend: bool = False
) -> Tuple[plt.Axes, matplotlib.legend.Legend]:
    """Plot a stacked horizontal bar chart to show category frequency.
    Works for boolean and categorical features.

    Args:
        data (pd.Series): category frequencies with category names as index
        colors (list): list of colors in a valid matplotlib format
        hide_legend (bool): if true, the legend is omitted

    Returns:
        ax: Stacked bar plot (matplotlib.axes)
        legend: Legend handler (matplotlib)
    """
    # Use the pd.Series indices as category names
    labels = data.index.values.astype(str)

    # Plot
    _, ax = plt.subplots(figsize=(7, 2))
    ax.axis("off")

    ax.set_xlim(0, np.sum(data))
    ax.set_ylim(0.4, 1.6)

    starts = 0
    for x, label, color in zip(data, labels, colors):
        # Add a rectangle to the stacked barh chart
        rects = ax.barh(y=1, width=x, height=1, left=starts, label=label, color=color)

        # Label color depends on the darkness of the rectangle
        r, g, b, _ = rects[0].get_facecolor()
        text_color = "white" if r * g * b < 0.5 else "darkgrey"

        # If the new bar is big enough write the label
        pc_of_total = x / data.sum() * 100
        # Requires matplotlib >= 3.4.0
        if pc_of_total > 8 and hasattr(ax, "bar_label"):
            display_txt = f"{pc_of_total:.1f}%\n({x})"
            ax.bar_label(
                rects,
                labels=[display_txt],
                label_type="center",
                color=text_color,
                fontsize="x-large",
                fontweight="bold",
            )

        starts += x

    legend = None
    if not hide_legend:
        legend = ax.legend(
            ncol=1, bbox_to_anchor=(0, 0), fontsize="xx-large", loc="upper left"
        )

    return ax, legend


def _plot_pie_chart(
    data: pd.Series, colors: List, hide_legend: bool = False
) -> Tuple[plt.Axes, matplotlib.legend.Legend]:
    """Plot a pie chart to show category frequency.
    Works for boolean and categorical features.

    Args:
        data (pd.Series): category frequencies with category names as index
        colors (list): list of colors in a valid matplotlib format
        hide_legend (bool): if true, the legend is omitted

    Returns:
        ax: pie chart (matplotlib.axes)
        legend: Legend handler (matplotlib)
    """

    def make_autopct(values: pd.Series) -> Callable:
        def my_autopct(pct: float) -> str:
            total = np.sum(values)
            val = int(round(pct * total / 100.0))
            return f"{pct:.1f}%  ({val:d})"

        return my_autopct

    _, ax = plt.subplots(figsize=(4, 4))
    wedges, _, _ = plt.pie(
        data,
        autopct=make_autopct(data),
        textprops={"color": "w"},
        colors=colors,
    )

    legend = None
    if not hide_legend:
        legend = plt.legend(
            wedges,
            data.index.values,
            fontsize="large",
            bbox_to_anchor=(0, 0),
            loc="upper left",
        )

    return ax, legend


@manage_matplotlib_context()
def cat_frequency_plot(
    config: Settings,
    data: pd.Series,
) -> str:
    """Generate category frequency plot to show category frequency.
    Works for boolean and categorical features.

    Modify colors by setting 'config.plot.cat_freq.colors' to a
    list of valid matplotib colors:
    https://matplotlib.org/stable/tutorials/colors/colors.html

    Args:
        config (Settings): a profile report config
        data (pd.Series): category frequencies with category names as index

    Returns:
        str: encoded category frequency plot encoded
    """
    # Get colors, if not defined, use matplotlib defaults
    colors = config.plot.cat_freq.colors
    if colors is None:
        # Get matplotlib defaults
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # If there are more categories than colors, loop through the colors again
    if len(colors) < len(data):
        multiplier = int(len(data) / len(colors)) + 1
        colors = multiplier * colors  # repeat colors as required
        colors = colors[0 : len(data)]  # select the exact number of colors required

    # Create the plot
    plot_type = config.plot.cat_freq.type
    if plot_type == "bar":
        if isinstance(data, list):
            for v in data:
                plot, legend = _plot_stacked_barh(
                    v, colors, hide_legend=config.vars.cat.redact
                )
        else:
            plot, legend = _plot_stacked_barh(
                data, colors, hide_legend=config.vars.cat.redact
            )

    elif plot_type == "pie":
        plot, legend = _plot_pie_chart(data, colors, hide_legend=config.vars.cat.redact)

    else:
        msg = (
            f"'{plot_type}' is not a valid plot type! "
            "Expected values are ['bar', 'pie']"
        )
        msg
        raise ValueError(msg)

    return plot_360_n0sc0pe(
        config,
        bbox_extra_artists=[] if legend is None else [legend],
        bbox_inches="tight",
    )


def create_comparison_color_list(config: Settings) -> List[str]:
    colors = config.html.style.primary_colors
    labels = config.html.style._labels

    if colors < labels:
        init = colors[0]
        end = colors[1] if len(colors) >= 2 else "#000000"
        cmap = LinearSegmentedColormap.from_list("ts_leg", [init, end], len(labels))
        colors = [rgb2hex(cmap(i)) for i in range(cmap.N)]
    return colors


def _plot_timeseries(
    config: Settings,
    series: Union[list, pd.Series],
    figsize: tuple = (6, 4),
) -> matplotlib.figure.Figure:
    """Plot an line plot from the data and return the AxesSubplot object.
    Args:
        series: The data to plot
        figsize: The size of the figure (width, height) in inches, default (6,4)
    Returns:
        The TimeSeries lineplot.
    """
    fig = plt.figure(figsize=figsize)
    plot = fig.add_subplot(111)

    if isinstance(series, list):
        labels = config.html.style._labels
        colors = create_comparison_color_list(config)

        for serie, color, label in zip(series, colors, labels):
            serie.plot(color=color, label=label)

    else:
        series.plot(color=config.html.style.primary_colors[0])

    return plot


@manage_matplotlib_context()
def mini_ts_plot(config: Settings, series: Union[list, pd.Series]) -> str:
    """Plot an time-series plot of the data.
    Args:
      series: The data to plot.
    Returns:
      The resulting timeseries plot encoded as a string.
    """
    plot = _plot_timeseries(config, series, figsize=(3, 2.25))
    plot.xaxis.set_tick_params(rotation=45)
    plt.rc("ytick", labelsize=3)

    for tick in plot.xaxis.get_major_ticks():
        if isinstance(series.index, pd.DatetimeIndex):
            tick.label1.set_fontsize(6)
        else:
            tick.label1.set_fontsize(8)
    plot.figure.tight_layout()
    return plot_360_n0sc0pe(config)


def _get_ts_lag(config: Settings, series: pd.Series) -> int:
    lag = config.vars.timeseries.pacf_acf_lag
    max_lag_size = (len(series) // 2) - 1
    return np.min([lag, max_lag_size])


def _plot_acf_pacf(
    config: Settings, series: pd.Series, figsize: tuple = (15, 5)
) -> str:
    color = config.html.style.primary_colors[0]

    lag = _get_ts_lag(config, series)
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    plot_acf(
        series.dropna(),
        lags=lag,
        ax=axes[0],
        title="ACF",
        fft=True,
        color=color,
        vlines_kwargs={"colors": color},
    )
    plot_pacf(
        series.dropna(),
        lags=lag,
        ax=axes[1],
        title="PACF",
        method="ywm",
        color=color,
        vlines_kwargs={"colors": color},
    )

    for ax in axes:
        for item in ax.collections:
            if type(item) == PolyCollection:
                item.set_facecolor(color)

    return plot_360_n0sc0pe(config)


def _plot_acf_pacf_comparison(
    config: Settings, series: List[pd.Series], figsize: tuple = (15, 5)
) -> str:
    colors = config.html.style.primary_colors
    n_labels = len(config.html.style._labels)
    colors = create_comparison_color_list(config)

    _, axes = plt.subplots(nrows=n_labels, ncols=2, figsize=figsize)
    is_first = True
    for serie, (acf_axis, pacf_axis), color in zip(series, axes, colors):
        lag = _get_ts_lag(config, serie)

        plot_acf(
            serie.dropna(),
            lags=lag,
            ax=acf_axis,
            title="ACF" if is_first else "",
            fft=True,
            color=color,
            vlines_kwargs={"colors": color},
        )
        plot_pacf(
            serie.dropna(),
            lags=lag,
            ax=pacf_axis,
            title="PACF" if is_first else "",
            method="ywm",
            color=color,
            vlines_kwargs={"colors": color},
        )
        is_first = False

    for row, color in zip(axes, colors):
        for ax in row:
            for item in ax.collections:
                if isinstance(item, PolyCollection):
                    item.set_facecolor(color)

    return plot_360_n0sc0pe(config)


@manage_matplotlib_context()
def plot_acf_pacf(
    config: Settings, series: Union[list, pd.Series], figsize: tuple = (15, 5)
) -> str:
    if isinstance(series, list):
        return _plot_acf_pacf_comparison(config, series, figsize)
    else:
        return _plot_acf_pacf(config, series, figsize)


def _prepare_heatmap_data(
    dataframe: pd.DataFrame,
    entity_column: str,
    sortby: Optional[Union[str, list]] = None,
    max_entities: int = 5,
    selected_entities: Optional[List[str]] = None,
) -> pd.DataFrame:
    if sortby is None:
        sortbykey = "_index"
        df = dataframe[entity_column].copy().reset_index()
        df.columns = [sortbykey, entity_column]

    else:
        if isinstance(sortby, str):
            sortby = [sortby]
        cols = [entity_column, *sortby]
        df = dataframe[cols].copy()
        sortbykey = sortby[0]

    if df[sortbykey].dtype == "O":
        try:
            df[sortbykey] = pd.to_datetime(df[sortbykey])
        except Exception as ex:
            raise ValueError(
                f"column {sortbykey} dtype {df[sortbykey].dtype} is not supported."
            ) from ex
    nbins = np.min([50, df[sortbykey].nunique()])

    df["__bins"] = pd.cut(
        df[sortbykey], bins=nbins, include_lowest=True, labels=range(nbins)
    )

    df = df.groupby([entity_column, "__bins"])[sortbykey].count()
    df = df.reset_index().pivot(entity_column, "__bins", sortbykey).T
    if selected_entities:
        df = df[selected_entities].T
    else:
        df = df.T[:max_entities]

    return df


def _create_timeseries_heatmap(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 5),
    color: str = "#337ab7",
) -> plt.Axes:
    _, ax = plt.subplots(figsize=figsize)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "report", ["white", color], N=64
    )
    pc = ax.pcolormesh(df, edgecolors=ax.get_facecolor(), linewidth=0.25, cmap=cmap)
    pc.set_clim(0, np.nanmax(df))
    ax.set_yticks([x + 0.5 for x in range(len(df))])
    ax.set_yticklabels(df.index)
    ax.set_xticks([])
    ax.set_xlabel("Time")
    ax.invert_yaxis()
    return ax


@typechecked
def timeseries_heatmap(
    dataframe: pd.DataFrame,
    entity_column: str,
    sortby: Optional[Union[str, list]] = None,
    max_entities: int = 5,
    selected_entities: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 5),
    color: str = "#337ab7",
) -> plt.Axes:
    """Generate a multi entity timeseries heatmap based on a pandas DataFrame.

    Args:
        dataframe: the pandas DataFrame
        entity_column: name of the entities column
        sortby: column that define the timesteps (only dates and numerical variables are supported)
        max_entities: max entities that will be displayed
        selected_entities: Optional list of entities to be displayed (overules max_entities)
        figsize: The size of the figure (width, height) in inches, default (10,5)
        color: the primary color, default '#337ab7'
    Returns:
        The TimeSeries heatmap.
    """
    df = _prepare_heatmap_data(
        dataframe, entity_column, sortby, max_entities, selected_entities
    )
    ax = _create_timeseries_heatmap(df, figsize, color)
    ax.set_aspect(1)
    return ax


def _set_visibility(
    axis: matplotlib.axis.Axis, tick_mark: str = "none"
) -> matplotlib.axis.Axis:
    for anchor in ["top", "right", "bottom", "left"]:
        axis.spines[anchor].set_visible(False)
    axis.xaxis.set_ticks_position(tick_mark)
    axis.yaxis.set_ticks_position(tick_mark)
    return axis


def missing_bar(
    notnull_counts: pd.Series,
    nrows: int,
    figsize: Tuple[float, float] = (25, 10),
    fontsize: float = 16,
    labels: bool = True,
    color: Tuple[float, ...] = (0.41, 0.41, 0.41),
    label_rotation: int = 45,
) -> matplotlib.axis.Axis:
    """
    A bar chart visualization of the missing data.

    Inspired by https://github.com/ResidentMario/missingno

    Args:
        notnull_counts: Number of nonnull values per column.
        nrows: Number of rows in the dataframe.
        figsize: The size of the figure to display.
        fontsize: The figure's font size. This default to 16.
        labels: Whether or not to display the column names. Would need to be turned off on particularly large
            displays. Defaults to True.
        color: The color of the filled columns. Default to the RGB multiple `(0.25, 0.25, 0.25)`.
        label_rotation: What angle to rotate the text labels to. Defaults to 45 degrees.
    Returns:
        The plot axis.
    """
    percentage = notnull_counts / nrows

    if len(notnull_counts) <= 50:
        ax0 = percentage.plot.bar(figsize=figsize, fontsize=fontsize, color=color)
        ax0.set_xticklabels(
            ax0.get_xticklabels(),
            ha="right",
            fontsize=fontsize,
            rotation=label_rotation,
        )

        ax1 = ax0.twiny()
        ax1.set_xticks(ax0.get_xticks())
        ax1.set_xlim(ax0.get_xlim())
        ax1.set_xticklabels(
            notnull_counts, ha="left", fontsize=fontsize, rotation=label_rotation
        )
    else:
        ax0 = percentage.plot.barh(figsize=figsize, fontsize=fontsize, color=color)
        ylabels = ax0.get_yticklabels() if labels else []
        ax0.set_yticklabels(ylabels, fontsize=fontsize)

        ax1 = ax0.twinx()
        ax1.set_yticks(ax0.get_yticks())
        ax1.set_ylim(ax0.get_ylim())
        ax1.set_yticklabels(notnull_counts, fontsize=fontsize)

    for ax in [ax0, ax1]:
        ax = _set_visibility(ax)

    return ax0


def missing_matrix(
    notnull: Any,
    columns: List[str],
    height: int,
    figsize: Tuple[float, float] = (25, 10),
    color: Tuple[float, ...] = (0.41, 0.41, 0.41),
    fontsize: float = 16,
    labels: bool = True,
    label_rotation: int = 45,
) -> matplotlib.axis.Axis:
    """
    A matrix visualization of missing data.

    Inspired by https://github.com/ResidentMario/missingno

    Args:
        notnull: Missing data indicator matrix.
        columns: List of column names.
        height: Number of rows in the dataframe.
        figsize: The size of the figure to display.
        fontsize: The figure's font size. Default to 16.
        labels: Whether or not to display the column names when there is more than 50 columns.
        label_rotation: What angle to rotate the text labels to. Defaults to 45 degrees.
        color: The color of the filled columns. Default is `(0.41, 0.41, 0.41)`.
    Returns:
        The plot axis.
    """
    width = len(columns)
    missing_grid = np.zeros((height, width, 3), dtype=np.float32)

    missing_grid[notnull] = color
    missing_grid[~notnull] = [1, 1, 1]

    _, ax = plt.subplots(1, 1, figsize=figsize)

    # Create the missing matrix plot.
    ax.imshow(missing_grid, interpolation="none")
    ax.set_aspect("auto")
    ax.grid(False)
    ax.xaxis.tick_top()

    ha = "left"
    ax.set_xticks(list(range(0, width)))
    ax.set_xticklabels(columns, rotation=label_rotation, ha=ha, fontsize=fontsize)
    ax.set_yticks([0, height - 1])
    ax.set_yticklabels([1, height], fontsize=fontsize)

    separators = [x + 0.5 for x in range(0, width - 1)]
    for point in separators:
        ax.axvline(point, linestyle="-", color="white")

    if not labels and width > 50:
        ax.set_xticklabels([])

    ax = _set_visibility(ax)
    return ax


def missing_heatmap(
    corr_mat: Any,
    mask: Any,
    figsize: Tuple[float, float] = (20, 12),
    fontsize: float = 16,
    labels: bool = True,
    label_rotation: int = 45,
    cmap: str = "RdBu",
    normalized_cmap: bool = True,
    cbar: bool = True,
    ax: matplotlib.axis.Axis = None,
) -> matplotlib.axis.Axis:
    """
    Presents a `seaborn` heatmap visualization of missing data correlation.
    Note that this visualization has no special support for large datasets.

    Inspired by https://github.com/ResidentMario/missingno

    Args:
        corr_mat: correlation matrix.
        mask: Upper-triangle mask.
        figsize: The size of the figure to display. Defaults to (20, 12).
        fontsize: The figure's font size.
        labels: Whether or not to label each matrix entry with its correlation (default is True).
        label_rotation: What angle to rotate the text labels to. Defaults to 45 degrees.
        cmap: Which colormap to use. Defaults to `RdBu`.
        normalized_cmap: Use a normalized colormap threshold or not. Defaults to True
    Returns:
        The plot axis.
    """
    _, ax = plt.subplots(1, 1, figsize=figsize)
    norm_args = {"vmin": -1, "vmax": 1} if normalized_cmap else {}

    if labels:
        sns.heatmap(
            corr_mat,
            mask=mask,
            cmap=cmap,
            ax=ax,
            cbar=cbar,
            annot=True,
            annot_kws={"size": fontsize - 2},
            **norm_args,
        )
    else:
        sns.heatmap(corr_mat, mask=mask, cmap=cmap, ax=ax, cbar=cbar, **norm_args)

    # Apply visual corrections and modifications.
    ax.xaxis.tick_bottom()
    ax.set_xticklabels(
        ax.xaxis.get_majorticklabels(),
        rotation=label_rotation,
        ha="right",
        fontsize=fontsize,
    )
    ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=0, fontsize=fontsize)
    ax = _set_visibility(ax)
    ax.patch.set_visible(False)

    for text in ax.texts:
        t = float(text.get_text())
        if 0.95 <= t < 1:
            text.set_text("<1")
        elif -1 < t <= -0.95:
            text.set_text(">-1")
        elif t == 1:
            text.set_text("1")
        elif t == -1:
            text.set_text("-1")
        elif -0.05 < t < 0.05:
            text.set_text("")
        else:
            text.set_text(round(t, 1))

    return ax
