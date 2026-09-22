"""
Generates the result tables (and a small macro file with the headline numbers) for the thesis'
evaluation chapter (07_evaluation.tex) from the final-run outputs, so that finishing more runs
only means re-running this script -- nothing in the chapter has to be edited by hand.

Inputs (all read-only):
  data/combined/03_integration/autoencoders/{002um,008um}/loso_token_log_*.csv   retrieval, leakage
  data/combined/03_integration/autoencoders/final_loso_clustering_quality_smoothed.csv
  data/combined/03_integration/autoencoders/final_loso_disease_signal.csv
  data/combined/03_integration/autoencoders/final_loso_gene_expression_probe.csv
  data/combined/03_integration/autoencoders/final_loso_cluster_annotation.csv
  02_baselines/<baseline>/outputs/<baseline>_clustering_quality.csv              baselines

Outputs, one file per table, in --out-dir (default: the thesis' content/generated/ directory):
  07_tab_*.tex   \\input-able tables in the thesis' house style (rwthblau rules, striped rows)
  07_results_macros.tex   \\newcommand{\\Res...} for the numbers quoted in running text
Coverage is computed, not assumed: every caption states how many of the 9 LOSO folds each
configuration has, and configurations without any finished folds are printed as "pending".

Adding a baseline: add an entry to BASELINES (name, CSV path). The CSV needs the columns of
spagcn_clustering_quality.csv: sample, label_kind ("pred" / "refined_pred"), reference_kind,
ari_vs_label_transfer, nmi_vs_label_transfer; a missing file is rendered as a "pending" row.

Usage (from 04_eval/):
    python make_thesis_tables.py
    python make_thesis_tables.py --out-dir /tmp/tables
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "data/combined/03_integration/autoencoders"
DEFAULT_OUT = Path("/home/charlottewolf/Nextcloud/Arbeit UKA/MA/03_Ausarbeitung/thesis/content/generated")

SLIDES = ["CRC_P1", "CRC_P11mm", "CRC_P11mm_2_cut", "CRC_P2", "CRC_P3_SMAP",
          "CRC_P4_SMAP", "CRC_P5", "Normal_P3", "Normal_P5"]
NORMAL_SLIDES = ["Normal_P3", "Normal_P5"]
NO_REFERENCE = {"CRC_P2"}  # label-transfer reference slide: no independent reference labels to score against

UNIT_DIR = {"nuclei": ("002um", "", "cap300000"), "bins": ("008um", "_bins", "cap610000")}
VARIANT_TAG = {"svg": "", "colon_xenium_panel": "_panel_colon_xenium",
               "svg_semantic_excl_raw_0.2": "_anchorsim0.2_simsrc_raw"}
MAIN_CONFIGS = [("nuclei", "svg"), ("nuclei", "colon_xenium_panel"), ("bins", "svg"), ("bins", "colon_xenium_panel")]
ARM_CONFIGS = [("nuclei", "svg_semantic_excl_raw_0.2"), ("bins", "svg_semantic_excl_raw_0.2")]
ALL_CONFIGS = MAIN_CONFIGS + ARM_CONFIGS
LABEL = {"svg": "SVG", "colon_xenium_panel": "Xenium panel", "svg_semantic_excl_raw_0.2": "SVG + semantic excl."}
# short forms for tables where cfg_label()'s full text (up to "Bins, SVG + semantic excl.") is what pushes the
# table past \textwidth; the caption/prose elsewhere always uses the long form, only these dense tables abbreviate.
UNIT_SHORT = {"nuclei": "Nuc", "bins": "Bin"}
LABEL_SHORT = {"svg": "SVG", "colon_xenium_panel": "Xenium", "svg_semantic_excl_raw_0.2": "SVG+excl."}
CELLTYPE_SHORT = {  # display-only abbreviations for the widest assigned_cell_type strings; data itself is untouched
    "Epithelial (general)": "Epithelial (gen.)", "Myeloid / Macrophage": "Myeloid/Macro.",
    "Fibroblast / Stromal": "Fibroblast/Stromal", "Colonocyte / Enterocyte": "Colonocyte/Entero.",
}


def cfg_label_short(cfg):
    return f"{UNIT_SHORT[cfg[0]]}, {LABEL_SHORT[cfg[1]]}"

BASELINES = {  # name -> outputs CSV; STAGATE / SpatialFusion appear as "pending" until their CSV exists
    "SpaGCN": REPO_ROOT / "02_baselines/spagcn/outputs/spagcn_clustering_quality.csv",
    "STAGATE": REPO_ROOT / "02_baselines/stagate/outputs/stagate_clustering_quality.csv",
    "SpatialFusion": REPO_ROOT / "02_baselines/spatialfusion/outputs/spatialfusion_clustering_quality.csv",
}
EMBEDDINGS = ["z_trans", "z_token", "z_joint"]
EMB_LABEL = {"z_trans": r"$\mathbf{z}^{\mathrm{trans}}$", "z_token": r"$\mathbf{z}^{\mathrm{token}}$",
             "z_joint": r"$[\mathbf{z}^{\mathrm{trans}};\mathbf{z}^{\mathrm{token}}]$"}


# ---- formatting helpers ---------------------------------------------------

def cfg_label(cfg):
    return f"{cfg[0].capitalize()}, {LABEL[cfg[1]]}"


def tex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def f3(x, digits=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def pending():
    return r"\textcolor{red}{pending}"


def head(cells):
    return " & ".join(rf"\textcolor{{rwthblau100}}{{\textbf{{{c}}}}}" for c in cells) + r" \\"


def table(colspec, header_rows, rows, caption_short, caption, label, size=r"\footnotesize", resize=False):
    """rows: list of cell lists; a row may be the string "\\midrule" for a separator."""
    out = [r"\begin{table}[htbp]", r"    \centering", f"    {size}"]
    if resize:
        out.append(r"    \resizebox{\textwidth}{!}{%")
    out += [rf"    \begin{{tabular}}{{{colspec}}}", r"        \arrayrulecolor{rwthblau100}\toprule"]
    out += [f"        {h}" for h in header_rows]
    out.append(r"        \arrayrulecolor{rwthblau100}\midrule")
    stripe = 0
    for r in rows:
        if r == r"\midrule":
            out.append(r"        \arrayrulecolor{rwthblau100}\midrule")
            stripe = 0
            continue
        prefix = r"\rowcolor{rwthblau10} " if stripe % 2 else ""
        out.append(f"        {prefix}" + " & ".join(r) + r" \\")
        stripe += 1
    out.append(r"        \arrayrulecolor{rwthblau100}\bottomrule")
    out.append(r"    \end{tabular}" + ("}" if resize else ""))
    out += [rf"    \caption[{caption_short}]", rf"            {{{caption}}}", rf"    \label{{{label}}}", r"\end{table}"]
    return "\n".join(out) + "\n"


def write(out_dir, name, content):
    (out_dir / name).write_text(content)
    print(f"  wrote {name}")


MACROS = {}


def macro(name, value):
    MACROS[name] = value


def bold_best(vals, higher=True, digits=3):
    """Format floats and bold the best one(s); None/NaN -> '--'."""
    ok = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    best = (max(ok) if higher else min(ok)) if ok else None
    return ["--" if x is None or (isinstance(x, float) and np.isnan(x))
            else (r"\textbf{" + f"{x:.{digits}f}" + "}" if x == best else f"{x:.{digits}f}") for x in vals]


# ---- data loading -----------------------------------------------------------

def run_tag(unit, variant):
    _, prefix, cap = UNIT_DIR[unit]
    anchor = VARIANT_TAG[variant] if variant == "svg_semantic_excl_raw_0.2" else ""
    panel = VARIANT_TAG[variant] if variant == "colon_xenium_panel" else ""
    return f"{prefix}_edrop0.1_gdrop0.1{anchor}_wd1e-05{panel}_{cap}_oversample_normal"


def load_retrieval():
    rows = []
    for unit, variant in ALL_CONFIGS:
        for slide in SLIDES:
            p = BASE / UNIT_DIR[unit][0] / f"loso_token_log_{slide}{run_tag(unit, variant)}.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p).iloc[-1]
            rows.append(dict(unit=unit, variant=variant, slide=slide,
                             held_cluster=d["held_out_cell_cluster_top1_retrieval"],
                             held_top1=d["held_out_cell_top1_retrieval"],
                             train_cluster=d["train_cell_cluster_top1_retrieval"],
                             leakage=d["combined_token_leakage_mean"], leak_chance=d["combined_token_leakage_chance"],
                             frac_masked=d["final_local_frac_masked"]))
    return pd.DataFrame(rows)


def cover(df, cfg, col="slide"):
    unit, variant = cfg
    return df[(df.unit == unit) & (df.variant == variant)][col].nunique() if len(df) else 0


def coverage_note(dfs):
    """'Nuclei/SVG 9/9, Bins/SVG 2/9, ...' -- appended to captions so they never go stale."""
    parts = []
    for cfg in ALL_CONFIGS:
        n = cover(dfs, cfg)
        parts.append(f"{cfg_label(cfg)} {n}/{len(SLIDES)}")
    return "Finished folds: " + "; ".join(parts) + "."


# ---- tables -------------------------------------------------------------------

def tab_retrieval(ret, out):
    hdr = [head(["Held-out slide"] + [cfg_label(c) for c in MAIN_CONFIGS])]
    rows = []
    for s in SLIDES:
        vals = []
        for cfg in MAIN_CONFIGS:
            v = ret[(ret.unit == cfg[0]) & (ret.variant == cfg[1]) & (ret.slide == s)]
            vals.append(float(v.held_cluster.iloc[0]) if len(v) else None)
        rows.append([tex(s)] + bold_best(vals))
    rows.append(r"\midrule")
    means, n_row = [], ["Folds finished"]
    for cfg in MAIN_CONFIGS:
        v = ret[(ret.unit == cfg[0]) & (ret.variant == cfg[1])]
        means.append(float(v.held_cluster.mean()) if len(v) else None)
        n_row.append(f"{len(v)}/{len(SLIDES)}")
    rows += [[r"\textbf{Mean}"] + [m if m != "--" else pending() for m in bold_best(means)], n_row]
    cap = ("Held-out $\\text{top1}_{\\text{cluster}}$ per \\gls{loso} fold; the best value per row is bold. ``--'': fold not finished, "
           "excluded from the mean. Not comparable across slides (reference cluster counts differ) or between units.")
    write(out, "07_tab_retrieval_per_slide.tex",
          table("lrrrr", hdr, rows, "Held-out cluster-level retrieval per LOSO fold.", cap, "tab:07_retrieval_per_slide"))


def tab_summary(ret, disease, out):
    hdr = [r"\textcolor{rwthblau100}{\textbf{Configuration}} & \textcolor{rwthblau100}{\textbf{Folds}} & "
           r"\multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{Held-out}}} & \textcolor{rwthblau100}{\textbf{In-sample}} & "
           r"\textcolor{rwthblau100}{\textbf{Slide\_ID}} \\",
           r"\textcolor{rwthblau100}{} & \textcolor{rwthblau100}{} & "
           r"\textcolor{rwthblau100}{$\text{top1}_{\text{cl}}$} & \textcolor{rwthblau100}{exact top-1} & "
           r"\textcolor{rwthblau100}{$\text{top1}_{\text{cl}}$} & \textcolor{rwthblau100}{leakage} \\"]
    stats = {}
    for cfg in ALL_CONFIGS:
        v = ret[(ret.unit == cfg[0]) & (ret.variant == cfg[1])]
        stats[cfg] = None if not len(v) else dict(n=len(v), c=v.held_cluster.mean(), e=v.held_top1.mean(),
                                                  t=v.train_cluster.mean(), l=v.leakage.mean())
    have = [c for c in ALL_CONFIGS if stats[c]]
    fc = dict(zip(have, bold_best([stats[c]["c"] for c in have])))
    fe = dict(zip(have, bold_best([stats[c]["e"] for c in have], digits=4)))
    fl = dict(zip(have, bold_best([stats[c]["l"] for c in have], higher=False)))
    rows = []
    for cfg in ALL_CONFIGS:
        st = stats[cfg]
        if not st:
            rows.append([cfg_label_short(cfg), "0", pending(), "", "", ""])
            continue
        rows.append([cfg_label_short(cfg), f"{st['n']}", fc[cfg], fe[cfg], f3(st["t"]), fl[cfg]])
        key = {"nuclei": "Nuc", "bins": "Bin"}[cfg[0]] + {"svg": "Svg", "colon_xenium_panel": "Panel",
                                                          "svg_semantic_excl_raw_0.2": "Sem"}[cfg[1]]
        macro(f"Res{key}ClusterTop", f3(st["c"]))
        macro(f"Res{key}ExactTop", f3(st["e"], 4))
        macro(f"Res{key}TrainCluster", f3(st["t"]))
        macro(f"Res{key}Leak", f3(st["l"]))
        macro(f"Res{key}Folds", str(st["n"]))
    chance = ret.leak_chance.dropna().iloc[0] if len(ret) else float("nan")
    cap = ("Summary of the final \\gls{loso} runs, mean over finished folds; the best value per column is bold (retrieval and leakage "
           "only; in-sample fit is not a quality measure). Nuc/Bin: nuclei/bins; excl.: semantic exclusion. "
           "$\\text{top1}_{\\text{cl}}$ is the primary retrieval metric. Slide\\_ID "
           f"leakage is held-out-vs-train linear-probe accuracy on the token embedding (chance $\\approx {chance:.2f}$, lower is better). "
           "Not directly comparable between units. " + coverage_note(ret))
    write(out, "07_tab_final_summary.tex",
          table("lrrrrr", hdr, rows, "Summary of the final LOSO runs.", cap, "tab:07_final_summary"))


def paired_macros(ret):
    """SVG vs. Xenium panel on the folds finished for BOTH (means of unequal fold sets aren't comparable)."""
    for unit, key in [("nuclei", "Nuc"), ("bins", "Bin")]:
        a = ret[(ret.unit == unit) & (ret.variant == "svg")].set_index("slide")
        b = ret[(ret.unit == unit) & (ret.variant == "colon_xenium_panel")].set_index("slide")
        common = a.index.intersection(b.index)
        if not len(common):
            continue
        macro(f"Res{key}PairedN", str(len(common)))
        macro(f"Res{key}PairedSvg", f3(a.held_cluster[common].mean()))
        macro(f"Res{key}PairedPanel", f3(b.held_cluster[common].mean()))
        macro(f"Res{key}PairedSvgWins", str(int((a.held_cluster[common] > b.held_cluster[common]).sum())))


def tab_semantic_arm(ret, out):
    hdr = [r"\textcolor{rwthblau100}{\textbf{Held-out slide}} & \multicolumn{3}{c}{\textcolor{rwthblau100}{\textbf{Nuclei}}} & "
           r"\multicolumn{3}{c}{\textcolor{rwthblau100}{\textbf{Bins}}} \\",
           r"\textcolor{rwthblau100}{} & " + " & ".join(rf"\textcolor{{rwthblau100}}{{{m}}}" for m in
                                                        [r"SVG", r"+ sem.\ excl.", r"$\Delta$"] * 2) + r" \\"]
    rows = []
    for s in SLIDES:
        row = [tex(s)]
        for unit in ["nuclei", "bins"]:
            a = ret[(ret.unit == unit) & (ret.variant == "svg") & (ret.slide == s)]
            b = ret[(ret.unit == unit) & (ret.variant == "svg_semantic_excl_raw_0.2") & (ret.slide == s)]
            va = a.held_cluster.iloc[0] if len(a) else None
            vb = b.held_cluster.iloc[0] if len(b) else None
            row += [f3(va), f3(vb), f"{vb - va:+.3f}" if va is not None and vb is not None else "--"]
        rows.append(row)
    rows.append(r"\midrule")
    mrow = [r"\textbf{Mean (paired folds)}"]
    for unit in ["nuclei", "bins"]:
        a = ret[(ret.unit == unit) & (ret.variant == "svg")].set_index("slide").held_cluster
        b = ret[(ret.unit == unit) & (ret.variant == "svg_semantic_excl_raw_0.2")].set_index("slide").held_cluster
        common = a.index.intersection(b.index)
        if len(common):
            mrow += [f3(a[common].mean()), f3(b[common].mean()), f"{(b[common] - a[common]).mean():+.3f}"]
            if unit == "nuclei":
                sub3 = [x for x in ["CRC_P2", "CRC_P4_SMAP", "Normal_P3"] if x in common]
                macro("ResSemDeltaSubset", f"{(b[sub3] - a[sub3]).mean():+.3f}")
                macro("ResSemDeltaNuc", f"{(b[common] - a[common]).mean():+.3f}")
                macro("ResSemPosNuc", str(int(((b[common] - a[common]) > 0).sum())))
                macro("ResSemNNuc", str(len(common)))
        else:
            mrow += [pending(), "", ""]
    rows.append(mrow)
    cap = ("Raw-source semantic exclusion (threshold 0.2) versus the plain configuration, held-out "
           "$\\text{top1}_{\\text{cluster}}$ per fold. The mean is over folds where both runs exist. "
           + coverage_note(ret))
    write(out, "07_tab_semantic_arm.tex",
          table("lrrrrrr", hdr, rows, "Semantic-exclusion comparison arm.", cap, "tab:07_semantic_arm"))


def load_quality():
    d = pd.read_csv(BASE / "final_loso_clustering_quality_smoothed.csv").rename(columns={"held_out_slide": "slide"})
    # number of Leiden clusters found lives only in the unsmoothed table (same partition as label_kind "raw").
    # Some rows (e.g. CRC_P2's ground-truth fold, merged in directly with all of evaluate_clustering_quality.py's
    # columns) may already carry their own n_clusters_found -- drop it here so the merge below is the one and only
    # source, instead of a silent pandas _x/_y suffix once both sides have the column.
    d = d.drop(columns="n_clusters_found", errors="ignore")
    n = (pd.read_csv(BASE / "final_loso_clustering_quality.csv").rename(columns={"held_out_slide": "slide"})
         [["unit", "variant", "slide", "embedding", "n_clusters_found"]].drop_duplicates())
    return d.merge(n, on=["unit", "variant", "slide", "embedding"], how="left")


def agg_slide_mean(d, keys):
    """Mean over folds of per-fold values (one row per slide+keys already)."""
    return d.groupby(keys)[["ari_vs_label_transfer", "nmi_vs_label_transfer", "silhouette"]].mean()


def tab_clustering(q, out):
    raw_ref = q[q.reference_kind == "raw"]
    body = raw_ref[raw_ref.label_kind == "raw"]
    hdr = [head(["Configuration", "Embedding", "Folds", "Clusters", "ARI", "NMI", "Silhouette"])]

    def build(embs):
        rows = []
        for cfg in ALL_CONFIGS:
            first = True
            for emb in embs:
                v = body[(body.unit == cfg[0]) & (body.variant == cfg[1]) & (body.embedding == emb)]
                label = cfg_label(cfg) if first else ""
                first = False
                if not len(v):
                    rows.append([label, EMB_LABEL[emb], "0", pending() if emb == embs[0] else "", "", "", ""])
                    continue
                rows.append([label, EMB_LABEL[emb], str(len(v)), f"{v.n_clusters_found.mean():.1f}",
                             f3(v.ari_vs_label_transfer.mean()), f3(v.nmi_vs_label_transfer.mean()), f3(v.silhouette.mean())])
            if len(embs) > 1:
                rows.append(r"\midrule")
        return rows[:-1] if len(embs) > 1 else rows

    gt_row = q[(q.unit == "nuclei") & (q.variant == "svg") & (q.embedding == "z_trans") &
               (q.label_kind == "raw") & (q.reference_kind == "gt")]
    if len(gt_row):
        macro("ResGtFixedPTwoAri", f3(gt_row.ari_vs_label_transfer.mean()))
        macro("ResGtFixedPTwoNmi", f3(gt_row.nmi_vs_label_transfer.mean()))
    cap = ("Leiden clustering (resolution 0.5) against the bin-level label-transfer reference (nearest $8\\,\\mu$m grid cell, raw, "
           "unrefined); mean over folds with a label-transfer reference. \\texttt{CRC\\_P2}, the classifier's own training slide, has "
           "none; its agreement with the pathologist annotation at this same resolution is reported in the text instead, and at "
           "matched cluster counts in Table~\\ref{tab:07_ground_truth_p2}. Clusters: mean Leiden count. Silhouette: 10\\,000-unit subsample.")
    write(out, "07_tab_clustering_quality_ztrans.tex",
          table("llrrrrr", hdr, build(["z_trans"]), "Clustering quality of the transcriptomic embedding.",
                cap + " " + coverage_note(raw_ref.assign(slide=raw_ref.slide)), "tab:07_clustering_ztrans"))
    write(out, "10_tab_clustering_quality_all_embeddings.tex",
          table("llrrrrr", hdr, build(EMBEDDINGS), "Clustering quality of all three embeddings.",
                cap + " " + coverage_note(raw_ref), "tab:10_clustering_all"))
    for cfg in ALL_CONFIGS:
        v = body[(body.unit == cfg[0]) & (body.variant == cfg[1]) & (body.embedding == "z_trans")]
        if len(v):
            key = {"nuclei": "Nuc", "bins": "Bin"}[cfg[0]] + {"svg": "Svg", "colon_xenium_panel": "Panel", "svg_semantic_excl_raw_0.2": "Sem"}[cfg[1]]
            macro(f"Res{key}Ari", f3(v.ari_vs_label_transfer.mean()))
            macro(f"Res{key}Nmi", f3(v.nmi_vs_label_transfer.mean()))
            macro(f"Res{key}Sil", f3(v.silhouette.mean()))


def tab_smoothing(q, out, ref_kind="raw", name="07_tab_smoothing_effect", label="tab:07_smoothing_effect", embs=("z_trans",)):
    d = q[(q.reference_kind == ref_kind) & (q.embedding.isin(embs))]
    modes = ["raw", "hard", "spagcn"]
    hdr = [r"\textcolor{rwthblau100}{\textbf{Configuration}} & " + " & ".join(
        rf"\multicolumn{{3}}{{c}}{{\textcolor{{rwthblau100}}{{\textbf{{{m}}}}}}}" for m in ["ARI", "NMI", "Silhouette"]) + r" \\",
           r"\textcolor{rwthblau100}{} & " + " & ".join(
        rf"\textcolor{{rwthblau100}}{{{m}}}" for m in ["none", "maj.", "SpaGCN"] * 3) + r" \\"]
    rows = []
    for cfg in ALL_CONFIGS:
        v = d[(d.unit == cfg[0]) & (d.variant == cfg[1])]
        if not len(v):
            rows.append([cfg_label(cfg), pending()] + [""] * 8)
            continue
        row = [cfg_label(cfg)]
        for metric in ["ari_vs_label_transfer", "nmi_vs_label_transfer", "silhouette"]:
            for m in modes:
                row.append(f3(v[v.label_kind == m][metric].mean()))
        rows.append(row)
        key = {"nuclei": "Nuc", "bins": "Bin"}[cfg[0]] + {"svg": "Svg", "colon_xenium_panel": "Panel", "svg_semantic_excl_raw_0.2": "Sem"}[cfg[1]]
        if ref_kind == "raw":
            for m, mn in zip(modes, ["None", "Hard", "Spagcn"]):
                macro(f"Res{key}Ari{mn}", f3(v[v.label_kind == m].ari_vs_label_transfer.mean()))
                macro(f"Res{key}Nmi{mn}", f3(v[v.label_kind == m].nmi_vs_label_transfer.mean()))
                macro(f"Res{key}Sil{mn}", f3(v[v.label_kind == m].silhouette.mean()))
    what = "raw" if ref_kind == "raw" else "spatially smoothed (majority vote on the nucleus/bin graph)"
    cap = (f"Effect of spatial refinement (\\texttt{{refine\\_domains}}: none, majority vote, SpaGCN rule) on the Leiden partition of "
           f"{'the transcriptomic embedding' if len(embs) == 1 else 'the embeddings'}, against the {what} label-transfer reference. "
           "Mean over folds with a reference. " + coverage_note(d.assign(slide=d.slide)))
    write(out, f"{name}.tex", table("l" + "r" * 9, hdr, rows, "Effect of spatial refinement.", cap, label))


def load_baseline(name):
    p = BASELINES[name]
    if not p.exists():
        return None
    d = pd.read_csv(p).rename(columns={"sample": "slide"})
    return d[d.reference_kind == "raw"]


def tab_baselines(q, out):
    """Matched-slide comparison: only slides every AVAILABLE baseline has results for (and that have a
    reference, and are in the LOSO roster), so every row is scored on the same slides."""
    baselines = {n: load_baseline(n) for n in BASELINES}
    slide_sets = [set(d.slide.unique()) for d in baselines.values() if d is not None]
    matched = set(SLIDES) - NO_REFERENCE
    for s in slide_sets:
        matched &= s
    matched = [s for s in SLIDES if s in matched]
    hdr = [r"\textcolor{rwthblau100}{\textbf{Method}} & " + " & ".join(
        rf"\multicolumn{{3}}{{c}}{{\textcolor{{rwthblau100}}{{\textbf{{{m}}}}}}}" for m in ["ARI", "NMI"]) + r" & \textcolor{rwthblau100}{\textbf{Folds}} \\",
           r"\textcolor{rwthblau100}{} & " + " & ".join(rf"\textcolor{{rwthblau100}}{{{m}}}" for m in ["none", "maj.", "refined"] * 2) + r" & \\"]
    rows = []
    for name, d in baselines.items():
        if d is None:
            rows.append([name, pending()] + [""] * 6)
            continue
        d = d[d.slide.isin(matched)]
        per = d.groupby(["slide", "label_kind"])[["ari_vs_label_transfer", "nmi_vs_label_transfer"]].mean().reset_index()
        row = [f"{name}"]
        for metric in ["ari_vs_label_transfer", "nmi_vs_label_transfer"]:
            row += [f3(per[per.label_kind == "pred"][metric].mean()), "--", f3(per[per.label_kind == "refined_pred"][metric].mean())]
        row.append(str(per.slide.nunique()))
        rows.append(row)
        if name == "STAGATE":
            macro("ResStagateAriNone", f3(per[per.label_kind == "pred"].ari_vs_label_transfer.mean()))
            macro("ResStagateNmiNone", f3(per[per.label_kind == "pred"].nmi_vs_label_transfer.mean()))
            macro("ResStagateFolds", str(per.slide.nunique()))
        if name == "SpatialFusion":
            macro("ResSpatialFusionAriNone", f3(per[per.label_kind == "pred"].ari_vs_label_transfer.mean()))
            macro("ResSpatialFusionNmiNone", f3(per[per.label_kind == "pred"].nmi_vs_label_transfer.mean()))
            macro("ResSpatialFusionFolds", str(per.slide.nunique()))
        if name == "SpaGCN":
            for lk, mn in [("pred", "None"), ("refined_pred", "Refined")]:
                macro(f"ResSpagcnAri{mn}", f3(per[per.label_kind == lk].ari_vs_label_transfer.mean()))
                macro(f"ResSpagcnNmi{mn}", f3(per[per.label_kind == lk].nmi_vs_label_transfer.mean()))
            macro("ResSpagcnFolds", str(per.slide.nunique()))
            macro("ResSpagcnKRange", f"{int(d.n_clusters.min())}--{int(d.n_clusters.max())}")
    rows.append(r"\midrule")
    sub = q[(q.reference_kind == "raw") & (q.embedding == "z_trans") & (q.slide.isin(matched))]
    sp = baselines["SpaGCN"]
    if sp is not None:  # per-slide: our nuclei/SVG partition vs. SpaGCN (mean over k), same slides
        sp_s = sp[sp.slide.isin(matched) & (sp.label_kind == "pred")].groupby("slide").ari_vs_label_transfer.mean()
        ours = sub[(sub.unit == "nuclei") & (sub.variant == "svg") & (sub.label_kind == "raw")].set_index("slide").ari_vs_label_transfer
        common = sp_s.index.intersection(ours.index)
        macro("ResOursBeatSpagcnAri", str(int((ours[common] > sp_s[common]).sum())))
        macro("ResOursVsSpagcnN", str(len(common)))
    stg = baselines.get("STAGATE")
    if stg is not None:
        st_s = stg[stg.slide.isin(matched) & (stg.label_kind == "pred")].groupby("slide").ari_vs_label_transfer.mean()
        common = st_s.index.intersection(ours.index)
        macro("ResOursBeatStagateAri", str(int((ours[common] > st_s[common]).sum())))
        macro("ResOursVsStagateN", str(len(common)))
    sf = baselines.get("SpatialFusion")
    if sf is not None:
        sf_s = sf[sf.slide.isin(matched) & (sf.label_kind == "pred")].groupby("slide").ari_vs_label_transfer.mean()
        common = sf_s.index.intersection(ours.index)
        macro("ResOursBeatSpatialFusionAri", str(int((ours[common] > sf_s[common]).sum())))
        macro("ResOursVsSpatialFusionN", str(len(common)))
    for cfg in ALL_CONFIGS:
        v = sub[(sub.unit == cfg[0]) & (sub.variant == cfg[1])]
        if not len(v):
            rows.append([f"Genomosaic: {cfg_label(cfg)}", pending()] + [""] * 6)
            continue
        row = [f"Genomosaic: {cfg_label(cfg)}"]
        for metric in ["ari_vs_label_transfer", "nmi_vs_label_transfer"]:
            row += [f3(v[v.label_kind == "raw"][metric].mean()), f3(v[v.label_kind == "hard"][metric].mean()),
                    f3(v[v.label_kind == "spagcn"][metric].mean())]
        row.append(str(v.slide.nunique()))
        rows.append(row)
        key = {"nuclei": "Nuc", "bins": "Bin"}[cfg[0]] + {"svg": "Svg", "colon_xenium_panel": "Panel", "svg_semantic_excl_raw_0.2": "Sem"}[cfg[1]]
        for m, mn in [("raw", "None"), ("hard", "Hard"), ("spagcn", "Rule")]:
            macro(f"ResMatched{key}Ari{mn}", f3(v[v.label_kind == m].ari_vs_label_transfer.mean()))
            macro(f"ResMatched{key}Nmi{mn}", f3(v[v.label_kind == m].nmi_vs_label_transfer.mean()))
        macro(f"ResMatched{key}N", str(v.slide.nunique()))
    matched_txt = ", ".join(tex(s) for s in matched)
    cap = ("Spatial-domain baselines against our Leiden partition of $\\mathbf{z}^{\\mathrm{trans}}$, scored against the raw nearest-grid "
           "label-transfer reference on the same slides (" + matched_txt + "). ``maj.'': majority-vote refinement (ours only); "
           "``refined'': each method's own refinement (SpaGCN: \\texttt{spg.refine}). SpaGCN needs $k$ in advance ($k=7,\\dots,10$, "
           "averaged), whereas our partition (about 14 clusters at the default resolution) chooses its own $k$, so this table is not "
           "cluster-count matched -- Table~\\ref{tab:07_baseline_kmatched} is.")
    write(out, "07_tab_baseline_comparison.tex",
          table("lrrrrrrr", hdr, rows, "Comparison with spatial-domain baselines.", cap, "tab:07_baseline_comparison"))
    macro("ResBaselineMatchedN", str(len(matched)))


def tab_expression(out):
    d = pd.read_csv(BASE / "final_loso_gene_expression_probe.csv")
    hdr = [r"\textcolor{rwthblau100}{\textbf{Configuration}} & \textcolor{rwthblau100}{\textbf{Folds}} & "
           r"\multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{Median Pearson $r$}}} & "
           r"\multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{Median $R^2$}}} & "
           r"\textcolor{rwthblau100}{\textbf{Genes $R^2{>}0$ (MLP)}} \\",
           r"\textcolor{rwthblau100}{} & \textcolor{rwthblau100}{} & \textcolor{rwthblau100}{Ridge} & \textcolor{rwthblau100}{MLP} & "
           r"\textcolor{rwthblau100}{Ridge} & \textcolor{rwthblau100}{MLP} & \textcolor{rwthblau100}{} \\"]
    rows = []
    for cfg in ALL_CONFIGS:
        v = d[(d.unit == cfg[0]) & (d.variant == cfg[1])]
        if not len(v):
            rows.append([cfg_label(cfg), "0", pending(), "", "", "", ""])
            continue
        r, m = v[v.probe == "ridge"], v[v.probe == "mlp"]
        rows.append([cfg_label(cfg), str(r.held_out_slide.nunique()), f3(r.median_pearson_r.mean()), f3(m.median_pearson_r.mean()),
                     f3(r.median_r2.mean(), 4), f3(m.median_r2.mean(), 4), f3(m.frac_genes_r2_pos.mean())])
        key = {"nuclei": "Nuc", "bins": "Bin"}[cfg[0]] + {"svg": "Svg", "colon_xenium_panel": "Panel", "svg_semantic_excl_raw_0.2": "Sem"}[cfg[1]]
        macro(f"Res{key}ProbeRidge", f3(r.median_pearson_r.mean()))
        macro(f"Res{key}ProbeMlp", f3(m.median_pearson_r.mean()))
        macro(f"Res{key}ProbeMlpRtwo", f3(m.median_r2.mean(), 4))
        macro(f"Res{key}ProbeFrac", f3(m.frac_genes_r2_pos.mean()))
    cov = "; ".join(f"{cfg_label(c)} {d[(d.unit == c[0]) & (d.variant == c[1])].held_out_slide.nunique()}/{len(SLIDES)}" for c in ALL_CONFIGS)
    cap = ("Gene-expression prediction from the frozen histology-token embedding ($\\mathbf{z}^{\\mathrm{token}}$): Ridge and MLP probes "
           "fit on the training slides, scored on the held-out slide (per-fold gene medians, averaged over folds, at most 100\\,000 "
           "units/slide). $R^2$ near or below 0: no better than a constant prediction. Genes differ between gene sets, so rows are not "
           "gene-for-gene comparable. Finished folds: " + cov + ".")
    write(out, "07_tab_expression_probe.tex",
          table("lrrrrrr", hdr, rows, "Gene-expression prediction from histology.", cap, "tab:07_expression_probe"))


def tab_disease(out):
    d = pd.read_csv(BASE / "final_loso_disease_signal.csv")
    hdr = [r"\textcolor{rwthblau100}{\textbf{Configuration}} & \textcolor{rwthblau100}{\textbf{Folds}} & "
           r"\multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{Train LOSO}}} & \textcolor{rwthblau100}{\textbf{Tumor}} & "
           r"\textcolor{rwthblau100}{\textbf{Normal\_P3}} & \textcolor{rwthblau100}{\textbf{Normal\_P5}} \\",
           r"\textcolor{rwthblau100}{} & \textcolor{rwthblau100}{} & \textcolor{rwthblau100}{Acc.} & \textcolor{rwthblau100}{F1} & "
           r"\textcolor{rwthblau100}{slides} & \textcolor{rwthblau100}{} & \textcolor{rwthblau100}{} \\"]
    rows = []
    for cfg in ALL_CONFIGS:
        v = d[(d.unit == cfg[0]) & (d.variant == cfg[1])]
        if not len(v):
            rows.append([cfg_label_short(cfg), "0", pending(), "", "", "", ""])
            continue
        tumor = v[v.held_out_true_label == 1]
        row = [cfg_label_short(cfg), str(len(v)), f3(v.train_loso_acc.mean()), f3(v.train_loso_f1.mean()), f3(tumor.held_out_acc.mean())]
        for s in NORMAL_SLIDES:
            x = v[v.held_out_slide == s]
            row.append(f3(x.held_out_acc.iloc[0]) if len(x) else "--")
        rows.append(row)
        key = {"nuclei": "Nuc", "bins": "Bin"}[cfg[0]] + {"svg": "Svg", "colon_xenium_panel": "Panel", "svg_semantic_excl_raw_0.2": "Sem"}[cfg[1]]
        macro(f"Res{key}DiseaseTumor", f3(tumor.held_out_acc.mean()))
        macro(f"Res{key}DiseaseTrainLoso", f3(v.train_loso_acc.mean()))
        for s in NORMAL_SLIDES:
            x = v[v.held_out_slide == s]
            if len(x):
                macro(f"Res{key}Disease{'NormalThree' if s == 'Normal_P3' else 'NormalFive'}", f3(x.held_out_acc.iloc[0]))
    cap = ("Tumor-versus-normal signal in the frozen $\\mathbf{z}^{\\mathrm{trans}}$ (logistic regression). Nuc/Bin: nuclei/bins; "
           "excl.: semantic exclusion. Train \\gls{loso}: "
           "leave-one-training-slide-out accuracy and macro F1. Held-out columns: fraction of the held-out slide's units classified as "
           "its true class -- mean over tumor folds, and the two normal slides individually (single slides, so read differences of a "
           "few hundredths as noise; macro F1 is capped near 0.5 here since each left-out training slide has only one class). "
           + coverage_note(d.rename(columns={"held_out_slide": "slide"})))
    write(out, "07_tab_disease_signal.tex",
          table("lrrrrrr", hdr, rows, "Disease signal in the embeddings.", cap, "tab:07_disease_signal"))


def tab_annotation(out):
    d = pd.read_csv(BASE / "final_loso_cluster_annotation.csv")
    cfgs = [c for c in ALL_CONFIGS if len(d[(d.unit == c[0]) & (d.variant == c[1])])]
    types = list(pd.read_csv(BASE / "final_loso_cluster_annotation.csv").assigned_cell_type.unique())
    shares = {}
    for cfg in cfgs:
        v = d[(d.unit == cfg[0]) & (d.variant == cfg[1])]
        per_slide = v.groupby(["held_out_slide", "assigned_cell_type"]).n_cells_in_cluster.sum().unstack(fill_value=0)
        frac = per_slide.div(per_slide.sum(axis=1), axis=0)
        shares[cfg] = frac.mean(axis=0) * 100
    order = sorted(types, key=lambda t: -np.mean([shares[c].get(t, 0.0) for c in cfgs]))
    # two header rows (unit, then variant) instead of one wide "Nuclei, SVG + semantic excl."-style column
    # header per config -- with up to 6 config columns and only 1-2-digit percentages in the body, a single
    # combined header row was what pushed this table off the page width.
    hdr = [head(["Assigned cell type"] + [UNIT_SHORT[c[0]] for c in cfgs]),
           head([""] + [LABEL_SHORT[c[1]] for c in cfgs])]
    rows = [[CELLTYPE_SHORT.get(t, tex(t))] + [f"{shares[c].get(t, 0.0):.1f}" for c in cfgs] for t in order]
    rows.append(r"\midrule")
    rows.append(["Folds"] + [str(d[(d.unit == c[0]) & (d.variant == c[1])].held_out_slide.nunique()) for c in cfgs])
    cap = ("Share (\\%) of units whose Leiden cluster is assigned to each canonical marker-set cell type, per held-out slide then "
           "averaged. Assignment is the arg-max of the mean marker-set score per cluster (hand-curated marker list) -- a coarse sanity "
           "check, not a validated cell-type call. Nuc/Bin: nuclei/bins; excl.: semantic exclusion.")
    write(out, "07_tab_cluster_annotation.tex",
          table("l" + "r" * len(cfgs), hdr, rows, "Marker-based cluster annotation.", cap, "tab:07_cluster_annotation"))


def tab_umi_cutoff(out):
    """UMI-cutoff sensitivity for every slide and unit (umi_cutoff_sensitivity.py)."""
    path = BASE / "umi_cutoff_sensitivity.csv"
    if not path.exists():
        print("  umi_cutoff_sensitivity.csv not found -- UMI table skipped")
        return
    d = pd.read_csv(path)
    hdr = [r"\textcolor{rwthblau100}{\textbf{Unit}} & \textcolor{rwthblau100}{\textbf{Slide}} & \textcolor{rwthblau100}{\textbf{Units}} & "
           r"\textcolor{rwthblau100}{\textbf{Median}} & \textcolor{rwthblau100}{\textbf{Fixed}} & "
           r"\multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{Knee}}} & \multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{MAD}}} \\",
           r"\textcolor{rwthblau100}{} & \textcolor{rwthblau100}{} & \textcolor{rwthblau100}{$\geq1$ UMI} & \textcolor{rwthblau100}{UMI} & "
           r"\textcolor{rwthblau100}{kept} & \textcolor{rwthblau100}{cutoff} & \textcolor{rwthblau100}{kept} & "
           r"\textcolor{rwthblau100}{cutoff} & \textcolor{rwthblau100}{kept} \\"]
    pct = lambda x: f"{100 * x:.1f}\\%"
    rows = []
    for unit in ["nuclei", "bins"]:
        order = ["CRC_P1", "CRC_P2", "CRC_P3_SMAP", "CRC_P4_SMAP", "CRC_P5", "Normal_P3", "Normal_P5", "CRC_P6_5mm", "CRC_P11mm", "CRC_P11mm_2_cut"]
        for slide in order:
            r = d[(d.unit == unit) & (d.slide == slide)]
            if not len(r):
                rows.append([unit if slide == order[0] else "", tex(slide), r"\multicolumn{7}{c}{not available$^{\mathrm{a}}$}"])
                continue
            r = r.iloc[0]
            rows.append([unit if slide == order[0] else "", tex(slide), f"{int(r.n_units):,}".replace(",", r"\,"), f"{r.median_umi:.0f}",
                         pct(r.frac_kept_fixed), f"{r.knee_cutoff:.0f}", pct(r.frac_kept_knee),
                         f"{r.mad_cutoff:.1f}", pct(r.frac_kept_mad)])
        sub = d[d.unit == unit]
        rows.append(["", "Median over slides", "", f"{sub.median_umi.median():.0f}", pct(sub.frac_kept_fixed.median()),
                     f"{sub.knee_cutoff.median():.0f}", pct(sub.frac_kept_knee.median()), f"{sub.mad_cutoff.median():.1f}", pct(sub.frac_kept_mad.median())])
        rows.append(r"\midrule")
    rows = rows[:-1]
    cap = ("UMI-cutoff sensitivity per slide and unit (nuclei, $8\\,\\mu$m bins) with at least one \\gls{umi}, before any quality filter "
           "(so counts differ from Table~\\ref{tab:07_trans_preprocessing}). Fixed: the cutoff used throughout (total counts $>80$); "
           "knee: elbow of the sorted-count curve; MAD: median $-3\\times$MAD of log counts; ``kept'': share of units above the cutoff. "
           "$^{\\mathrm{a}}$Its pre-filter bin count covers a smaller region than the trained model uses (Section~\\ref{sect:07_exp_umi}), "
           "so \\texttt{CRC\\_P11mm\\_2\\_cut} is omitted here.")
    write(out, "07_tab_umi_cutoff_sensitivity.tex",
          table("llrrrrrrr", hdr, rows, "UMI-cutoff sensitivity per slide and unit.", cap, "tab:07_umi_cutoff_sensitivity"))


def tab_reference_agreement(out):
    path = BASE / "reference_agreement_sr_vs_lt.csv"
    if not path.exists():
        print("  reference_agreement_sr_vs_lt.csv not found -- reference-agreement table skipped")
        return
    d = pd.read_csv(path)
    hdr = [r"\textcolor{rwthblau100}{\textbf{Slide}} & \textcolor{rwthblau100}{\textbf{SR clusters}} & "
           r"\multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{ARI}}} & \multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{NMI}}} & "
           r"\textcolor{rwthblau100}{\textbf{ARI, $K{=}8$}} \\",
           r"\textcolor{rwthblau100}{} & \textcolor{rwthblau100}{} & \textcolor{rwthblau100}{LT} & \textcolor{rwthblau100}{LTs} & "
           r"\textcolor{rwthblau100}{LT} & \textcolor{rwthblau100}{LTs} & \textcolor{rwthblau100}{LT} \\"]
    rows = [[tex(r.slide), str(int(r.k_graph)), f3(r.ari_graph_lt), f3(r.ari_graph_lts), f3(r.nmi_graph_lt), f3(r.nmi_graph_lts), f3(r.ari_km8_lt)]
            for r in d.itertuples()]
    rows.append(r"\midrule")
    rows.append([r"\textbf{Mean}", f"{d.k_graph.mean():.1f}"] + [r"\textbf{" + f3(d[c].mean()) + "}" for c in
                 ("ari_graph_lt", "ari_graph_lts", "nmi_graph_lt", "nmi_graph_lts", "ari_km8_lt")])
    macro("ResRefAgreeAriLt", f3(d.ari_graph_lt.mean())); macro("ResRefAgreeNmiLt", f3(d.nmi_graph_lt.mean()))
    macro("ResRefAgreeAriLts", f3(d.ari_graph_lts.mean())); macro("ResRefAgreeNmiLts", f3(d.nmi_graph_lts.mean()))
    macro("ResRefAgreeAriMin", f3(d.ari_graph_lt.min())); macro("ResRefAgreeAriMax", f3(d.ari_graph_lt.max()))
    cap = ("Agreement between label transfer (LT), its majority-vote-smoothed version (LTs) and SpaceRanger's expression-only clusters "
           "(SR: graph-based, number per slide given; last column: k-means at $K=8$), on the native $8\\,\\mu$m bins. \\texttt{CRC\\_P2} "
           "(no LT) and the two SMAP slides (no native SpaceRanger output) are absent.")
    write(out, "07_tab_reference_agreement.tex",
          table("lrrrrrr", hdr, rows, "Agreement between label transfer and SpaceRanger clusters.", cap, "tab:07_reference_agreement"))


def write_macros(out):
    lines = ["% Generated by 04_eval/make_thesis_tables.py -- do not edit by hand."]
    for k, v in sorted(MACROS.items()):
        lines.append(rf"\newcommand{{\{k}}}{{{v}}}")
    write(out, "07_results_macros.tex", "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="where the final_loso_*.csv files are read from (default: data/combined/03_integration/autoencoders)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    global BASE
    if args.results_dir is not None:
        BASE = args.results_dir

    ret = load_retrieval()
    q = load_quality()
    disease = pd.read_csv(BASE / "final_loso_disease_signal.csv")
    paired_macros(ret)
    tab_retrieval(ret, args.out_dir)
    tab_summary(ret, disease, args.out_dir)
    tab_semantic_arm(ret, args.out_dir)
    tab_clustering(q, args.out_dir)
    tab_smoothing(q, args.out_dir)
    tab_smoothing(q, args.out_dir, ref_kind="smoothed", name="10_tab_smoothing_effect_smoothed_reference",
                  label="tab:10_smoothing_smoothed_ref")
    tab_baselines(q, args.out_dir)
    tab_expression(args.out_dir)
    tab_disease(args.out_dir)
    tab_annotation(args.out_dir)
    tab_umi_cutoff(args.out_dir)
    tab_reference_agreement(args.out_dir)
    import sys
    import make_thesis_tables_extended as ext
    ext.run(sys.modules[__name__], args.out_dir, BASE)  # tables from submit_evaluate_extended.sbatch's outputs
    write_macros(args.out_dir)


if __name__ == "__main__":
    main()
