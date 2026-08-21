import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import anndata as ad
from matplotlib.collections import PatchCollection
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle


def plot_spatial_labels(
    adata,
    spatial_key: str = "spatial",
    label_key: str = "predicted_annotation_filtered",
    ax=None,
    point_size: float = 3,
    palette: dict | None = None,
    title: str | None = None,
    invert_y: bool = True,
    legend: bool = True,
    mode: str = "bin",
    patch_size: float = 224,
):
    """
    Plot a single sample's bins/cells or patches, colored by a categorical
    annotation column.

    Parameters
    ----------
    adata : AnnData
        Must contain 2D coordinates in adata.obsm[spatial_key] and a
        categorical (or castable-to-categorical) column adata.obs[label_key].
    spatial_key : str
        Key in adata.obsm holding an (n_obs, 2) array of [x, y] coordinates.
        For mode="bin" this is the point position (e.g. pxl_col_in_fullres,
        pxl_row_in_fullres); for mode="patch" it's each patch's top-left
        corner.
    label_key : str
        Column in adata.obs holding the labels/annotation/clusters to color by.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. A new figure/axis is created if not given.
    point_size : float
        Marker size passed to scatter. Only used when mode="bin".
    palette : dict, optional
        Mapping {label: color}. If not given, one is generated. Pass a
        shared palette when plotting multiple samples so colors match
        across panels (plot_spatial_labels_grid does this for you).
    title : str, optional
        Plot title. Defaults to label_key.
    invert_y : bool
        Pixel row coordinates increase downward; invert the y-axis so the
        plot matches the tissue image orientation.
    legend : bool
        Whether to draw the category legend on this axis.
    mode : str
        "bin" (default) draws each row as a small dot -- use for bin/cell
        level data, where rows are far denser than their real footprint.
        "patch" draws each row as a `patch_size` x `patch_size` square
        anchored at its (x, y) top-left corner, tiling into a consistent
        grid -- use for patch-level data (e.g. UNI histology patches),
        where each row IS a contiguous square region of the image.
    patch_size : float
        Side length of each square in data (pixel) units. Only used when
        mode="patch".

    Returns
    -------
    (ax, palette) : the axis drawn on, and the {label: color} mapping used
        (hand this back in as `palette=` to reuse consistent colors).
    """
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"'{spatial_key}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )
    if label_key not in adata.obs:
        raise KeyError(
            f"'{label_key}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )
    if mode not in ("bin", "patch"):
        raise ValueError(f"mode must be 'bin' or 'patch', got {mode!r}")

    coords = np.asarray(adata.obsm[spatial_key])[:, :2]
    labels = adata.obs[label_key].astype("category")
    categories = list(labels.cat.categories)

    if palette is None:
        cmap = plt.get_cmap("tab20" if len(categories) <= 20 else "gist_ncar")
        denom = max(len(categories) - 1, 1)
        colors = [cmap(i / denom) for i in range(len(categories))]
        palette = dict(zip(categories, colors))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    for cat in categories:
        mask = (labels == cat).values
        if mask.sum() == 0:
            continue
        color = palette.get(cat, "lightgrey")
        if mode == "bin":
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=point_size,
                color=color,
                label=str(cat),
                linewidths=0,
            )
        else:  # mode == "patch"
            rects = [Rectangle((x, y), patch_size, patch_size) for x, y in coords[mask]]
            ax.add_collection(
                PatchCollection(rects, facecolor=color, edgecolor="none", label=str(cat))
            )

    if mode == "patch":
        ax.autoscale_view()
    ax.set_aspect("equal")
    if invert_y:
        ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or label_key)

    if legend:
        ax.legend(
            markerscale=4,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=8,
            frameon=False,
        )

    return ax, palette


def plot_spatial_labels_grid(
    adata,
    slide_key: str = "Slide_ID",
    spatial_key: str = "spatial",
    label_key: str = "predicted_annotation_filtered",
    ncols: int = 4,
    point_size: float = 2,
    shared_palette: bool = True,
    figsize_per_panel: tuple = (4, 4),
    mode: str = "bin",
    patch_size: float = 224,
):
    """
    Facet plot_spatial_labels across every slide in adata.obs[slide_key],
    using one consistent palette across panels so a given label always
    gets the same color, with a single shared legend for the figure.

    Parameters mirror plot_spatial_labels (including mode/patch_size);
    adata.obs must additionally contain a `slide_key` column identifying
    which sample each row belongs to (e.g. "Slide_ID").

    Returns
    -------
    (fig, axes)
    """
    if slide_key not in adata.obs:
        raise KeyError(
            f"'{slide_key}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    slides = adata.obs[slide_key].astype("category").cat.categories.tolist()
    n = len(slides)
    ncols_eff = min(ncols, n)
    nrows = int(np.ceil(n / ncols_eff))

    fig, axes = plt.subplots(
        nrows, ncols_eff,
        figsize=(figsize_per_panel[0] * ncols_eff, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    # Build one shared palette from the full label set up front, so the
    # same annotation gets the same color on every slide's panel.
    palette = None
    if shared_palette:
        all_labels = list(adata.obs[label_key].astype("category").cat.categories)
        cmap = plt.get_cmap("tab20" if len(all_labels) <= 20 else "gist_ncar")
        denom = max(len(all_labels) - 1, 1)
        colors = [cmap(i / denom) for i in range(len(all_labels))]
        palette = dict(zip(all_labels, colors))

    legend_entries = {}
    for i, slide in enumerate(slides):
        sub = adata[adata.obs[slide_key] == slide]
        _, used_palette = plot_spatial_labels(
            sub,
            spatial_key=spatial_key,
            label_key=label_key,
            ax=axes_flat[i],
            point_size=point_size,
            palette=palette,
            title=str(slide),
            legend=False,
            mode=mode,
            patch_size=patch_size,
        )
        legend_entries.update(used_palette)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    legend_marker = "s" if mode == "patch" else "o"
    handles = [
        plt.Line2D([0], [0], marker=legend_marker, linestyle="", color=c, label=str(l))
        for l, c in legend_entries.items()
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=9,
        frameon=False,
        title=label_key,
    )
    fig.tight_layout()
    return fig, axes


def plot_distribution_grid(
    values_by_slide: dict,
    xlabel: str,
    title: str | None = None,
    ncols: int = 4,
    bins: int = 50,
    cutoff: float | None = None,
    log_x: bool = False,
    figsize_per_panel: tuple = (4, 3),
):
    """
    Histogram of a per-nucleus/per-cell quantity (e.g. UMI counts, area),
    one panel per slide.

    Parameters
    ----------
    values_by_slide : dict[str, array-like]
        {slide_id: 1D array of values}.
    xlabel : str
        X-axis label, shared across panels.
    title : str, optional
        Figure suptitle.
    ncols : int
        Panels per row.
    bins : int
        Histogram bin count.
    cutoff : float, optional
        If given, draws a vertical dashed line at this value and annotates
        the fraction of values above it -- e.g. to show where a filtering
        threshold falls relative to the distribution.
    log_x : bool
        Log-scale the x-axis. Most per-nucleus distributions (UMI counts,
        area) are heavily right-skewed.
    figsize_per_panel : tuple
        (width, height) per panel, scaled by the grid shape.

    Returns
    -------
    (fig, axes)
    """
    BASE_COLOR = "#3470a3"       # muted blue -- below cutoff (or all bars, if no cutoff)
    HIGHLIGHT_COLOR = "#e8871e"  # warm amber -- above cutoff (retained)

    slides = list(values_by_slide.keys())
    n = len(slides)
    ncols_eff = min(ncols, n)
    nrows = int(np.ceil(n / ncols_eff))

    fig, axes = plt.subplots(
        nrows, ncols_eff,
        figsize=(figsize_per_panel[0] * ncols_eff, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, slide in enumerate(slides):
        ax = axes_flat[i]
        values = np.asarray(values_by_slide[slide])
        n_dropped = 0
        if log_x:
            # Zero/negative values have no log-space bin edge -- matplotlib's
            # default linear-then-rescale-axis approach collapses the whole
            # plot around them. Bin only the strictly-positive values, with
            # bin edges spaced in log space to begin with.
            positive = values[values > 0]
            bin_edges = np.geomspace(positive.min(), positive.max(), bins + 1)
            _, edges, patches = ax.hist(positive, bins=bin_edges, edgecolor="none")
            ax.set_xscale("log")
            n_dropped = len(values) - len(positive)
        else:
            _, edges, patches = ax.hist(values, bins=bins, edgecolor="none")

        # Muted base color throughout; bars entirely past the cutoff get the
        # brighter highlight, so the retained region reads at a glance.
        for patch, left_edge in zip(patches, edges[:-1]):
            patch.set_facecolor(HIGHLIGHT_COLOR if cutoff is not None and left_edge >= cutoff else BASE_COLOR)

        # Subtle horizontal gridlines only, no box -- eyes calibrate bar
        # height off the grid instead of feeling boxed in.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.3)

        note_lines = []
        if cutoff is not None:
            ax.axvline(cutoff, color=HIGHLIGHT_COLOR, linestyle="--", linewidth=1)
            kept_frac = (values > cutoff).mean()
            note_lines.append(f"{kept_frac:.0%} > {cutoff:g}")
        if n_dropped:
            note_lines.append(f"{n_dropped} zero/negative excluded")
        if note_lines:
            ax.text(
                0.98, 0.95, "\n".join(note_lines),
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.75),
            )

        ax.set_title(str(slide))
        ax.set_ylabel("count")

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    # One shared x-axis label at the bottom instead of repeating it on every
    # panel -- all panels plot the same quantity.
    fig.supxlabel(xlabel)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_filter_funnel(counts_df, colors: list | None = None, ax=None, title: str | None = None):
    """
    Horizontal 100%-stacked bar chart of nucleus/cell counts per filtering
    stage, one bar per slide.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Indexed by slide, one column per MUTUALLY EXCLUSIVE stage (e.g.
        "remaining", "removed_umi_only", "removed_area_only",
        "removed_both") -- each row must sum to that slide's total, since
        every row is normalized to 100% before plotting. Overlapping stages
        (e.g. separate "removed_umi"/"removed_area" columns that double-count
        nuclei failing both) will not sum to 100% and should be split into
        exclusive categories first.
    colors : list[str], optional
        One color per column, in column order. Defaults to a 5-step blue
        sequential palette (darkest first), truncated/cycled to the number
        of columns.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. A new figure/axis is created if not given.
    title : str, optional
        Plot title.

    Returns
    -------
    ax
    """
    default_palette = ["#00549F", "#407FB7", "#8EBAE5", "#C7DDF2", "#E8F1FA"]
    if colors is None:
        colors = [default_palette[i % len(default_palette)] for i in range(counts_df.shape[1])]

    pct_df = counts_df.div(counts_df.sum(axis=1), axis=0) * 100
    slides = pct_df.index.astype(str)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 0.6 * len(pct_df) + 1.5))

    left = np.zeros(len(pct_df))
    for col, color in zip(pct_df.columns, colors):
        widths = pct_df[col].to_numpy()
        ax.barh(slides, widths, left=left, color=color, edgecolor="none", label=col)
        for y, (w, l) in enumerate(zip(widths, left)):
            if w >= 5:
                ax.text(l + w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=8, color="black")
        left += widths

    ax.set_xlim(0, 100)
    ax.set_xlabel("% of nuclei")
    ax.set_ylabel("Slide_ID")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title or "Nucleus counts per filtering stage")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.invert_yaxis()  # first slide at top
    return ax


def plot_raincloud_grid(
    values_by_group: dict,
    xlabel: str,
    title: str | None = None,
    colors: dict | None = None,
    figsize: tuple | None = None,
    jitter_n: int = 300,
    seed: int = 0,
    xlim: tuple | None = None,
):
    """
    Horizontal raincloud plot -- half-violin density ("cloud") + mini
    boxplot + jittered raw points ("rain") -- for each (group, stage) pair,
    one row per pair. Shows the full distribution shape (e.g. bimodality
    from a dead-cell population) that a bar chart of means would hide.

    Parameters
    ----------
    values_by_group : dict[str, dict[str, array-like]]
        {group_label: {stage_label: 1D array of values}}, e.g.
        {"CRC_P1": {"pre-filtering": [...], "post-filtering": [...]}, ...}.
        Rows are drawn group-by-group in dict order, with each group's
        stages kept together (e.g. pre directly above post).
    xlabel : str
        Shared x-axis label.
    title : str, optional
        Figure title.
    colors : dict[str, str], optional
        {stage_label: color}. Defaults to a small fixed palette assigned
        in stage-label order of first appearance.
    figsize : tuple, optional
        Defaults to a height scaled by the number of rows.
    jitter_n : int
        Max raw points drawn per row (subsampled if there are more, purely
        to keep the plot from getting too dense/slow to render).
    seed : int
        Random seed for jitter and subsampling.
    xlim : tuple, optional
        (left, right) view limits. Values/outliers beyond this are still
        used for the boxplot stats and KDE fit (nothing is dropped from the
        underlying data), just not shown -- useful when a few extreme
        outliers would otherwise squash the rest of the distribution into a
        sliver of the plot.

    Returns
    -------
    ax
    """
    from scipy.stats import gaussian_kde

    rng = np.random.default_rng(seed)
    rows = [
        (group, stage, np.asarray(values))
        for group, stages in values_by_group.items()
        for stage, values in stages.items()
    ]

    stage_labels = list(dict.fromkeys(stage for _, stage, _ in rows))
    if colors is None:
        default_colors = ["#B0413E", "#00549F", "#8EBAE5", "#E8871E"]
        colors = {s: default_colors[i % len(default_colors)] for i, s in enumerate(stage_labels)}

    n_rows = len(rows)
    if figsize is None:
        figsize = (9, 0.6 * n_rows + 1)
    fig, ax = plt.subplots(figsize=figsize)

    yticks, yticklabels = [], []
    for i, (group, stage, values) in enumerate(rows):
        y0 = n_rows - i  # top-to-bottom row order
        color = colors.get(stage, "grey")
        values = values[np.isfinite(values)]
        if len(values) < 2 or np.ptp(values) == 0:
            continue

        # Half-violin ("cloud"): KDE fit on the full data, but the grid only
        # spans the visible window (if capped) so resolution isn't wasted
        # sampling a long outlier tail that won't be shown anyway.
        kde = gaussian_kde(values)
        grid_max = min(values.max(), xlim[1]) if xlim and xlim[1] is not None else values.max()
        x_grid = np.linspace(values.min(), grid_max, 200)
        density = kde(x_grid)
        density = density / density.max() * 0.32  # scale to fit within the row band
        ax.fill_between(x_grid, y0, y0 + density, color=color, alpha=0.6, linewidth=0)

        # Mini boxplot, thin, just below the violin baseline.
        ax.boxplot(
            values, positions=[y0 - 0.08], vert=False, widths=0.12, manage_ticks=False,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.8, linewidth=0.8),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color="black", linewidth=0.8),
            capprops=dict(color="black", linewidth=0.8),
            flierprops=dict(marker=".", markersize=1, alpha=0.3),
        )

        # Jittered raw points ("rain"), below the box.
        sample = values if len(values) <= jitter_n else rng.choice(values, size=jitter_n, replace=False)
        jitter_y = y0 - 0.22 - rng.uniform(0, 0.14, size=len(sample))
        ax.scatter(sample, jitter_y, s=4, color=color, alpha=0.4, linewidths=0)

        yticks.append(y0)
        yticklabels.append(f"{group} — {stage}")

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.7) for c in colors.values()]
    ax.legend(handles, colors.keys(), bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    return ax


def plot_spatial_points(
    coords,
    ax=None,
    point_size: float = 1,
    color: str = "steelblue",
    title: str | None = None,
    invert_y: bool = True,
):
    """
    Plot spatial coordinates as same-colored dots, with no categorical
    labeling -- complements plot_spatial_labels for quick before/after
    comparisons where there's no annotation column to color by.

    Parameters
    ----------
    coords : array-like, shape (n, 2)
        [x, y] positions.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. A new figure/axis is created if not given.
    point_size : float
        Marker size passed to scatter.
    color : str
        Single color for every point.
    title : str, optional
        Plot title. Defaults to a point count.
    invert_y : bool
        Pixel row coordinates increase downward; invert the y-axis so the
        plot matches the tissue image orientation.

    Returns
    -------
    ax
    """
    coords = np.asarray(coords)[:, :2]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords[:, 0], coords[:, 1], s=point_size, color=color, linewidths=0)
    ax.set_aspect("equal")
    if invert_y:
        ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or f"{len(coords)} points")
    return ax


def plot_cluster_example_patches(
    adata,
    cluster_key: str,
    basedir: str | Path,
    umap_key: str = "X_umap",
    patch_dir_template: str = "{sample}/01_preprocessing/unimodel/patches",
    n_examples: int = 30,
    zoom: float = 0.15,
    ax=None,
    seed: int | None = None,
    title: str | None = None,
):
    """
    Overlay example patch-image thumbnails per cluster on a UMAP embedding,
    so clusters can be eyeballed (e.g. to spot background/no-tissue clusters
    before filtering them out). Works for a single sample's AnnData or a
    combined multi-sample AnnData alike -- each patch is located on disk via
    its own row's adata.obs["Slide_ID"] / adata.obs["Patch_ID"], so nothing
    special is needed for either case.

    Parameters
    ----------
    adata : AnnData
        Must contain a 2D embedding in adata.obsm[umap_key], a cluster
        column adata.obs[cluster_key], and adata.obs["Slide_ID"] /
        adata.obs["Patch_ID"] identifying each patch's PNG on disk.
    cluster_key : str
        Column in adata.obs holding cluster labels (e.g. "leiden_0.5").
    basedir : str or Path
        Root data directory; patch images are looked up at
        `basedir / patch_dir_template.format(sample=slide_id) / patch_id`.
    umap_key : str
        Key in adata.obsm holding the 2D embedding to scatter patches onto.
    patch_dir_template : str
        Per-sample patch directory, relative to basedir. "{sample}" is
        filled in with each row's Slide_ID.
    n_examples : int
        Max number of example patches drawn per cluster.
    zoom : float
        Thumbnail zoom level passed to OffsetImage.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. A new figure/axis is created if not given.
    seed : int, optional
        Random seed for which patches get sampled as examples per cluster.
    title : str, optional
        Plot title. Defaults to a description including cluster_key.

    Returns
    -------
    ax
    """
    if umap_key not in adata.obsm:
        raise KeyError(
            f"'{umap_key}' not found in adata.obsm. Available: {list(adata.obsm.keys())}"
        )
    if cluster_key not in adata.obs:
        raise KeyError(
            f"'{cluster_key}' not found in adata.obs. Available: {list(adata.obs.columns)}"
        )

    basedir = Path(basedir)
    umap_coords = np.asarray(adata.obsm[umap_key])
    # Plain numpy arrays, not Series, so integer indexing below is unambiguous
    # (no risk of accidentally reassigning/shadowing a Series with a scalar).
    clusters = adata.obs[cluster_key].astype(str).to_numpy()
    slide_ids = adata.obs["Slide_ID"].astype(str).to_numpy()
    patch_ids = adata.obs["Patch_ID"].astype(str).to_numpy()

    rng = random.Random(seed)

    if ax is None:
        _, ax = plt.subplots(figsize=(18, 18))
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c="lightgrey", s=5, linewidths=0)

    n_missing = 0
    for cluster in sorted(set(clusters), key=lambda c: (len(c), c)):
        cluster_indices = np.where(clusters == cluster)[0]
        selected = rng.sample(list(cluster_indices), min(n_examples, len(cluster_indices)))

        for i in selected:
            patch_path = basedir / patch_dir_template.format(sample=slide_ids[i]) / patch_ids[i]
            if not patch_path.exists():
                n_missing += 1
                continue
            img = OffsetImage(plt.imread(patch_path), zoom=zoom)
            ab = AnnotationBbox(img, umap_coords[i], frameon=False, box_alignment=(0.5, 0.5), pad=0)
            ax.add_artist(ab)

        avg_x, avg_y = umap_coords[cluster_indices].mean(axis=0)
        ax.annotate(
            f"Cluster {cluster}", (avg_x, avg_y), fontsize=16, ha="center", color="black", weight="bold"
        )

    if n_missing:
        print(f"Warning: {n_missing} example patch image(s) not found on disk.")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title or f"UMAP with example patches per {cluster_key}")
    return ax