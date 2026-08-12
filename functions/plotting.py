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