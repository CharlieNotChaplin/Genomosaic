"""
Extended thesis tables, generated from the outputs of submit_evaluate_extended.sbatch:
  final_loso_kmatched_ari.csv, final_loso_partition_agreement.csv, final_loso_baselines_ours_units.csv,
  final_loso_disease_signal_weighted.csv, final_loso_cluster_annotation_celltypist.csv (optional).
Called from make_thesis_tables.py (run that, not this file). Every table is skipped with a printed note if its input
file does not exist yet, so the script is safe to run before the cluster job has finished.

Conventions
  - "matched k": for our embeddings, the mean over all Leiden runs whose raw partition has 7..10 clusters (the
    baselines are averaged over k = 7..10 the same way). Resolution is never chosen with a reference.
  - "refined": majority-vote refinement (refine_domains mode "hard") of the raw partition.
  - References: LT = image-based label transfer, LTs = the same, majority-vote smoothed, SR = SpaceRanger graph
    clusters (expression-derived), GT = pathologist annotation (CRC_P2 only).
  - Baselines are scored on exactly OUR nuclei (final_loso_baselines_ours_units.csv), mean over their k = 7..10.
"""

import numpy as np
import pandas as pd

REFINED = "hard"
CFGS_MAIN = [("nuclei", "svg"), ("nuclei", "colon_xenium_panel"), ("bins", "svg"), ("bins", "colon_xenium_panel")]
CFGS_ALL = CFGS_MAIN + [("nuclei", "svg_semantic_excl_raw_0.2"), ("bins", "svg_semantic_excl_raw_0.2")]
BASELINE_NAMES = ["SpaGCN", "STAGATE", "SpatialFusion"]
mt = None  # make_thesis_tables module (set by run(); importing it here would create a second copy of its state)


def mean_or_nan(x):
    x = pd.Series(x).dropna()
    return float(x.mean()) if len(x) else np.nan


def bold_best(vals, higher=True):
    """Format floats, bold the best one(s); NaN -> '--'."""
    v = [x if x is not None and not (isinstance(x, float) and np.isnan(x)) else None for x in vals]
    ok = [x for x in v if x is not None]
    best = (max(ok) if higher else min(ok)) if ok else None
    return [("--" if x is None else (r"\textbf{" + f"{x:.3f}" + "}" if x == best else f"{x:.3f}")) for x in v]


def matched_means(km):
    """Mean over the runs with 7 <= k_raw <= 10, per (unit, variant, slide, embedding, partition)."""
    k = km["k_raw"] if "k_raw" in km.columns else km["k"]
    m = km[k.between(7, 10)]
    num = [c for c in m.columns if c.startswith(("ari_", "nmi_")) or c in ("silhouette", "chaos", "pas")]
    return m.groupby(["unit", "variant", "held_out_slide", "embedding", "partition"])[num].mean().reset_index()


def ours(km_m, unit, variant, emb, partition, col):
    d = km_m[(km_m.unit == unit) & (km_m.variant == variant) & (km_m.embedding == emb) & (km_m.partition == partition)]
    return d.set_index("held_out_slide")[col].dropna()


def load(base, name):
    """Reads <name> and any shard files (<stem>.shard*of*.csv) written by parallel runs, concatenated."""
    files = sorted(base.glob(name.replace(".csv", "*.csv")))
    if not files:
        print(f"  [extended] {name} not found -- tables that need it are skipped")
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def baseline_per_slide(bo, method, kind, col):
    d = bo[(bo.method == method) & (bo.label_kind == kind)]
    return d.groupby("held_out_slide")[col].mean().dropna()


# ---- per-sample ARI ------------------------------------------------------------------------------

def tab_ari_per_sample(km_m, bo, out, unit="nuclei", variant="svg", appendix=False):
    cfg = (unit, variant)
    cols = [("ari_lt", "LT"), ("ari_sr", "SR"), ("ari_gt", "annotation")]
    hdr = [r"\textcolor{rwthblau100}{\textbf{Slide}} & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{\textcolor{{rwthblau100}}{{\textbf{{ARI vs.\ {n}}}}}}}" for _, n in cols) +
        r" & \multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{Baselines (LT)}}} \\",
           r"\textcolor{rwthblau100}{} & " + " & ".join(rf"\textcolor{{rwthblau100}}{{{m}}}" for m in ["raw", "ref."] * 3) +
           r" & \textcolor{rwthblau100}{SpaGCN} & \textcolor{rwthblau100}{STAGATE} \\"]
    rows, allvals = [], {c: [] for c in range(8)}
    sp = baseline_per_slide(bo, "SpaGCN", "pred", "ari_lt") if bo is not None else pd.Series(dtype=float)
    st = baseline_per_slide(bo, "STAGATE", "pred", "ari_lt") if bo is not None else pd.Series(dtype=float)
    for s in mt.SLIDES:
        vals = []
        for col, _ in cols:
            for part in ("raw", REFINED):
                x = ours(km_m, unit, variant, "z_trans", part, col)
                vals.append(float(x[s]) if s in x.index else np.nan)
        vals += [float(sp[s]) if s in sp.index else np.nan, float(st[s]) if s in st.index else np.nan]
        rows.append([mt.tex(s)] + [mt.f3(v) for v in vals])
        for i, v in enumerate(vals):
            allvals[i].append(v)
    rows.append(r"\midrule")
    rows.append([r"\textbf{Mean}"] + [r"\textbf{" + mt.f3(mean_or_nan(allvals[i])) + "}" if not np.isnan(mean_or_nan(allvals[i])) else "--" for i in range(8)])
    rows.append(["Median"] + [mt.f3(float(np.nanmedian(allvals[i])) if np.isfinite(allvals[i]).any() else np.nan) for i in range(8)])
    cap = (f"Per-slide agreement of the Leiden partition of $\\mathbf{{z}}^{{\\mathrm{{trans}}}}$ ({mt.cfg_label(cfg)}, mean over runs with 7--10 "
           "clusters) with the image-based label transfer (LT), SpaceRanger's graph clusters (SR) and, for CRC\\_P2 only, the pathologist "
           "annotation, before (raw) and after majority-vote refinement (ref.). The last two columns give the baselines' \\gls{ari} against LT on the "
           "same nuclei (mean over $k=7,\\dots,10$). ``--'': no reference or no result for that slide (CRC\\_P2 has no LT reference; SR does not exist for "
           "the two SMAP slides). Slides with low values for the baselines as well point to the reference, not the embedding.")
    name = f"10_tab_ari_per_sample_{unit}_{variant}" if appendix else "07_tab_ari_per_sample"
    mt.write(out, name + ".tex", mt.table("l" + "r" * 8, hdr, rows, "Per-slide ARI.", cap,
                                         "tab:" + name[:2] + "_ari_per_sample" + (f"_{unit}_{variant}" if appendix else ""),
                                         size=r"\scriptsize", resize=True))


def tab_ari_per_sample_embeddings(km_m, out, unit="nuclei", variant="svg"):
    embs = ["z_trans", "z_token", "z_joint"]
    hdr = [r"\textcolor{rwthblau100}{\textbf{Slide}} & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{\textcolor{{rwthblau100}}{{\textbf{{{mt.EMB_LABEL[e]}}}}}}}" for e in embs) + r" \\",
           r"\textcolor{rwthblau100}{} & " + " & ".join(rf"\textcolor{{rwthblau100}}{{{m}}}" for m in ["LT", "SR"] * 3) + r" \\"]
    rows = []
    for s in mt.SLIDES:
        vals = []
        for e in embs:
            for col in ("ari_lt", "ari_sr"):
                x = ours(km_m, unit, variant, e, "raw", col)
                vals.append(float(x[s]) if s in x.index else np.nan)
        rows.append([mt.tex(s)] + [mt.f3(v) for v in vals])
    cap = ("\\gls{ari} of the raw Leiden partition (mean over runs with 7--10 clusters) of the transcriptomic, the histology and the "
           f"concatenated embedding ({mt.cfg_label((unit, variant))}) against label transfer (LT) and SpaceRanger (SR). The label-transfer "
           "reference is derived from histology tokens, which favours $\\mathbf{z}^{\\mathrm{token}}$ on LT; SR is derived from expression.")
    mt.write(out, "10_tab_ari_per_sample_embeddings.tex", mt.table("l" + "r" * 6, hdr, rows, "Per-slide ARI for the three embeddings.", cap,
                                                                 "tab:10_ari_per_sample_embeddings", size=r"\footnotesize"))


# ---- baseline comparison on our nuclei -------------------------------------------------------------

def method_rows(km_m, bo, unit_cfgs, slides, emb="z_trans"):
    """[(label, {col: value})] with raw / refined columns for baselines and ours, mean over `slides`."""
    def m(series):
        return mean_or_nan(series[series.index.isin(slides)])
    rows = []
    for name in BASELINE_NAMES:
        has = bo is not None and (bo.method == name).any()
        if not has:
            rows.append((name, None))
            continue
        r = {}
        for col in ("ari_lt", "ari_sr", "nmi_lt", "nmi_sr", "chaos", "pas"):
            r[(col, "raw")] = m(baseline_per_slide(bo, name, "pred", col))
            r[(col, "ref")] = m(baseline_per_slide(bo, name, "refined_pred", col)) if (bo[(bo.method == name)].label_kind == "refined_pred").any() else np.nan
        r["n"] = len(slides_with(bo, name, slides))
        rows.append((name, r))
    for cfg in unit_cfgs:
        r = {}
        for col in ("ari_lt", "ari_sr", "nmi_lt", "nmi_sr", "chaos", "pas", "silhouette"):
            r[(col, "raw")] = m(ours(km_m, *cfg, emb, "raw", col))
            r[(col, "ref")] = m(ours(km_m, *cfg, emb, REFINED, col))
        r["n"] = len([s for s in slides if s in ours(km_m, *cfg, emb, "raw", "ari_lt").index])
        rows.append((f"Ours: {mt.cfg_label(cfg)}", r if r["n"] else None))
    return rows


def slides_with(bo, name, slides):
    return sorted(set(bo[bo.method == name].held_out_slide) & set(slides))


def matched_slides(bo, km_m):
    """Slides with LT and SR references where every AVAILABLE baseline has results."""
    have = set(km_m[km_m.ari_lt.notna() & km_m.ari_sr.notna()].held_out_slide)
    if bo is not None:
        for name in BASELINE_NAMES:
            if (bo.method == name).any():
                have &= set(bo[bo.method == name].held_out_slide)
    return [s for s in mt.SLIDES if s in have]


def tab_baseline_kmatched(km_m, bo, out):
    if bo is None:
        return
    slides = matched_slides(bo, km_m)
    hdr = [r"\textcolor{rwthblau100}{\textbf{Method}} & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{\textcolor{{rwthblau100}}{{\textbf{{{n}}}}}}}" for n in ["\\gls{ari} vs.\\ LT", "\\gls{ari} vs.\\ SR", "\\gls{nmi} vs.\\ SR"]) +
        r" & \textcolor{rwthblau100}{\textbf{Folds}} \\",
           r"\textcolor{rwthblau100}{} & " + " & ".join(rf"\textcolor{{rwthblau100}}{{{m}}}" for m in ["raw", "ref."] * 3) + r" & \\"]
    rows = []
    for label, r in method_rows(km_m, bo, CFGS_ALL, slides):
        if r is None:
            rows.append([mt.tex(label), mt.pending()] + [""] * 6)
        else:
            rows.append([mt.tex(label)] + [mt.f3(r[(c, k)]) for c in ("ari_lt", "ari_sr", "nmi_sr") for k in ("raw", "ref")] + [str(r["n"])])
            if label == "Ours: Nuclei, SVG":
                for c, nm in (("ari_lt", "ResKmNucSvgAriLt"), ("ari_sr", "ResKmNucSvgAriSr"), ("nmi_sr", "ResKmNucSvgNmiSr")):
                    mt.macro(nm, mt.f3(r[(c, "raw")]))
                    mt.macro(nm + "Ref", mt.f3(r[(c, "ref")]))
            if label in BASELINE_NAMES:
                for c, nm in (("ari_lt", "AriLt"), ("ari_sr", "AriSr")):
                    mt.macro(f"ResKm{label}{nm}", mt.f3(r[(c, "raw")]))
    cap = ("Cluster-count-matched comparison with the spatial-domain baselines, all on the same nuclei and the same slides ("
           + ", ".join(mt.tex(s) for s in slides) + "). Our rows are the mean over the Leiden runs with 7--10 clusters, the baselines the mean over "
           "$k=7,\\dots,10$; ``ref.'' is majority-vote refinement for ours and the method's own refinement (SpaGCN only) for the baselines. "
           "LT: image-based label transfer (derived from histology); SR: SpaceRanger graph clusters (derived from expression). "
           "Neither is ground truth, the two agree with each other only at \\gls{ari} $\\approx0.17$ on average. Folds: slides averaged.")
    mt.write(out, "07_tab_baseline_comparison_kmatched.tex",
             mt.table("lrrrrrrr", hdr, rows, "Cluster-count-matched baseline comparison.", cap, "tab:07_baseline_kmatched", size=r"\footnotesize"))
    # label-free metrics
    hdr2 = [r"\textcolor{rwthblau100}{\textbf{Method}} & \multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{PAS} $\downarrow$}} & "
            r"\multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{CHAOS} $\downarrow$}} & \multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{Silhouette}}} \\",
            r"\textcolor{rwthblau100}{} & " + " & ".join(rf"\textcolor{{rwthblau100}}{{{m}}}" for m in ["raw", "ref."] * 3) + r" \\"]
    rows2 = []
    for label, r in method_rows(km_m, bo, CFGS_ALL, slides):
        if r is None:
            rows2.append([mt.tex(label), mt.pending()] + [""] * 5)
            continue
        sil = [mt.f3(r.get(("silhouette", k), np.nan)) if label not in BASELINE_NAMES else "--" for k in ("raw", "ref")]
        rows2.append([mt.tex(label)] + [mt.f3(r[(c, k)]) for c in ("pas", "chaos") for k in ("raw", "ref")] + sil)
    cap2 = ("Label-free spatial metrics on the same nuclei and slides as Table~\\ref{tab:07_baseline_kmatched}. PAS: share of units whose label differs "
            "from at least 6 of their 10 nearest spatial neighbours; CHAOS: mean distance to the nearest unit of the same cluster (pixels). Both reward "
            "spatial smoothness and do not need a reference, but they cannot tell a correct smooth partition from a wrong one. Silhouette is computed in "
            "the embedding that was clustered and does not exist for the baselines.")
    mt.write(out, "07_tab_label_free.tex", mt.table("lrrrrrr", hdr2, rows2, "Label-free spatial metrics.", cap2, "tab:07_label_free", size=r"\footnotesize"))


def tab_ground_truth(km_m, bo, out):
    slide = "CRC_P2"
    hdr = [r"\textcolor{rwthblau100}{\textbf{Method}} & \multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{\gls{ari}}}} & "
           r"\multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{\gls{nmi}}}} \\",
           r"\textcolor{rwthblau100}{} & " + " & ".join(rf"\textcolor{{rwthblau100}}{{{m}}}" for m in ["raw", "ref."] * 2) + r" \\"]
    rows = []
    for name in BASELINE_NAMES:
        if bo is None or not ((bo.method == name) & (bo.held_out_slide == slide)).any():
            rows.append([name, mt.pending(), "", "", ""])
            continue
        row = [name]
        for col in ("ari_gt", "nmi_gt"):
            for kind in ("pred", "refined_pred"):
                s = baseline_per_slide(bo, name, kind, col)
                row.append(mt.f3(float(s[slide])) if slide in s.index else "--")
        rows.append(row)
    rows.append(r"\midrule")
    for cfg in CFGS_MAIN[:2] + CFGS_ALL[4:5]:
        for emb in ("z_trans", "z_token", "z_joint"):
            row = [f"Ours: {mt.cfg_label(cfg)}, {mt.EMB_LABEL[emb]}"]
            got = False
            for col in ("ari_gt", "nmi_gt"):
                for part in ("raw", REFINED):
                    x = ours(km_m, *cfg, emb, part, col)
                    row.append(mt.f3(float(x[slide])) if slide in x.index else "--")
                    got |= slide in x.index
            if got:
                rows.append(row)
                if cfg == ("nuclei", "svg") and emb == "z_trans":
                    mt.macro("ResGtNucSvgAri", row[1]); mt.macro("ResGtNucSvgAriRef", row[2])
    if len(rows) <= 4:
        return
    cap = ("The one slide with real ground truth: agreement with the pathologist annotation of \\texttt{CRC\\_P2} (six base tissue classes, dominant "
           "component per $8\\,\\mu$m grid cell), for the held-out embedding of that \\gls{loso} fold and for the baselines, on the same nuclei. "
           "Ours: mean over Leiden runs with 7--10 clusters; baselines: mean over $k=7,\\dots,10$. A single slide, so differences of a few hundredths are noise.")
    mt.write(out, "07_tab_ground_truth_p2.tex", mt.table("lrrrr", hdr, rows, "Agreement with the pathologist annotation on CRC\\_P2.", cap,
                                                         "tab:07_ground_truth_p2", size=r"\footnotesize"))


# ---- sensitivity, agreement, appendix ---------------------------------------------------------------

def tab_ari_vs_k(km, out, unit="nuclei", variant="svg"):
    bins = [(2, 3), (4, 6), (7, 10), (11, 15), (16, 10_000)]
    d = km[(km.unit == unit) & (km.variant == variant) & (km.partition == "raw") & km.ari_lt.notna()]
    if not len(d):
        return
    hdr = [mt.head(["Clusters $k$"] + [mt.EMB_LABEL[e] for e in ("z_trans", "z_token", "z_joint")])]
    rows = []
    for lo, hi in bins:
        row = [f"{lo}--{hi}" if hi < 1000 else f"$\\geq{lo}$"]
        for e in ("z_trans", "z_token", "z_joint"):
            x = d[(d.embedding == e) & d.k.between(lo, hi)]
            row.append(mt.f3(x.groupby("held_out_slide").ari_lt.mean().mean()) if len(x) else "--")
        rows.append(row)
    cap = ("Mean \\gls{ari} against label transfer as a function of the number of Leiden clusters (each slide's runs averaged first, then slides), "
           f"{mt.cfg_label((unit, variant))}. The agreement falls steeply with $k$ because the reference has few dominant classes, which is why "
           "methods are only compared at the same cluster counts. No resolution is chosen with the reference.")
    mt.write(out, "10_tab_ari_vs_k.tex", mt.table("lrrr", hdr, rows, "ARI as a function of the number of clusters.", cap, "tab:10_ari_vs_k"))


def tab_partition_agreement(agree, out):
    if agree is None or not len(agree):
        return
    hdr = [mt.head(["Configuration", "Pair", "\\gls{ari}", "\\gls{nmi}", "Homogeneity", "Completeness"])]
    rows = []
    for cfg in CFGS_ALL:
        d = agree[(agree.unit == cfg[0]) & (agree.variant == cfg[1])]
        for (a, b), g in d.groupby(["emb_a", "emb_b"]):
            rows.append([mt.cfg_label(cfg), f"{mt.EMB_LABEL[a]} / {mt.EMB_LABEL[b]}", mt.f3(g.ari.mean()), mt.f3(g.nmi.mean()),
                         mt.f3(g.homogeneity_of_b_given_a.mean()), mt.f3(g.completeness_of_b_given_a.mean())])
    if not rows:
        return
    cap = ("Agreement between the Leiden partitions (about 8 clusters) of the different embeddings on the same held-out nuclei, mean over slides. "
           "Homogeneity: are the clusters of the second embedding pure with respect to the first; completeness: does the second keep the "
           "clusters of the first together. A joint partition that is complete and pure with respect to $\\mathbf{z}^{\\mathrm{trans}}$ finds no new "
           "structure; low homogeneity means it splits the transcriptomic clusters, i.e.\\ histology adds a distinction.")
    mt.write(out, "07_tab_partition_agreement.tex", mt.table("llrrrr", hdr, rows, "Agreement between embedding partitions.", cap,
                                                            "tab:07_partition_agreement", size=r"\footnotesize"))


def tab_disease_weighted(dw, out):
    if dw is None or not len(dw):
        return
    hdr = [r"\textcolor{rwthblau100}{\textbf{Configuration}} & \textcolor{rwthblau100}{\textbf{Embedding}} & "
           r"\multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{Tumor slides}}} & \multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{Normal\_P3}}} & "
           r"\multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{Normal\_P5}}} \\",
           r"\textcolor{rwthblau100}{} & \textcolor{rwthblau100}{} & " + " & ".join(rf"\textcolor{{rwthblau100}}{{{m}}}" for m in ["plain", "balanced"] * 3) + r" \\"]
    rows = []
    for cfg in [c for c in CFGS_ALL if c[0] == "nuclei"] + [c for c in CFGS_ALL if c[0] == "bins"]:
        for emb in sorted(dw.embedding.unique()):
            d = dw[(dw.unit == cfg[0]) & (dw.variant == cfg[1]) & (dw.embedding == emb)]
            if not len(d):
                continue
            row = [mt.cfg_label(cfg), mt.EMB_LABEL[emb]]
            for sel in ("tumor", "Normal_P3", "Normal_P5"):
                for cw in ("none", "balanced"):
                    x = d[(d.class_weight == cw) & ((d.held_out_true_label == 1) if sel == "tumor" else (d.held_out_slide == sel))]
                    row.append(mt.f3(x.held_out_acc.mean()) if len(x) else "--")
            rows.append(row)
            if cfg == ("nuclei", "svg") and emb == "z_trans":
                mt.macro("ResWeightedNormalThreePlain", row[4]); mt.macro("ResWeightedNormalThreeBal", row[5])
                mt.macro("ResWeightedNormalFivePlain", row[6]); mt.macro("ResWeightedNormalFiveBal", row[7])
    if not rows:
        return
    cap = ("Tumor-versus-normal probe on frozen embeddings, without and with balanced class weights (the training pool holds one normal slide against eight tumor "
           "slides when a normal slide is held out). Values: fraction of the held-out slide's units classified as its true class (tumor slides: mean over folds). "
           "Units are subsampled per slide. With two normal slides in the cohort, the normal columns are single-slide observations.")
    mt.write(out, "07_tab_disease_signal_weighted.tex", mt.table("llrrrrrr", hdr, rows, "Disease signal with and without class weighting.", cap,
                                                                "tab:07_disease_weighted", size=r"\footnotesize"))


def tab_celltypist(ct, out):
    if ct is None or not len(ct):
        return
    cfgs = [c for c in CFGS_ALL if len(ct[(ct.unit == c[0]) & (ct.variant == c[1])])]
    shares = {}
    for cfg in cfgs:
        v = ct[(ct.unit == cfg[0]) & (ct.variant == cfg[1])]
        per = v.groupby(["held_out_slide", "assigned_cell_type"]).n_cells_annotated.sum().unstack(fill_value=0)
        shares[cfg] = (per.div(per.sum(axis=1), axis=0).mean(axis=0) * 100)
    types = sorted({t for s in shares.values() for t in s.index}, key=lambda t: -np.mean([shares[c].get(t, 0.0) for c in cfgs]))[:15]
    hdr = [mt.head(["Assigned cell type (CellTypist)"] + [mt.cfg_label(c) for c in cfgs])]
    rows = [[mt.tex(t)] + [f"{shares[c].get(t, 0.0):.1f}" for c in cfgs] for t in types]
    cap = ("Share (\\%) of units in clusters that CellTypist (majority vote over our Leiden clusters) assigns to each cell type, per slide, then averaged "
           "over slides; the 15 most frequent types. Predictions on sparse nucleus-level counts are noisy; this complements, and does not replace, the marker-based annotation.")
    mt.write(out, "07_tab_cluster_annotation_celltypist.tex", mt.table("l" + "r" * len(cfgs), hdr, rows, "CellTypist annotation.", cap,
                                                                      "tab:07_celltypist", size=r"\footnotesize"))


def tab_long_summary(km_m, bo, out):
    """Appendix: one long table. Columns: ARI vs smoothed LT (raw|ref) and vs SR (raw|ref), NMI vs smoothed LT and vs SR
    (refined partitions only), silhouette (ours, z_trans). Blocks: all slides with SR; slides shared with each baseline."""
    hdr = [r"\textcolor{rwthblau100}{\textbf{Method}} & \multicolumn{4}{c}{\textcolor{rwthblau100}{\textbf{\gls{ari}}}} & "
           r"\multicolumn{2}{c}{\textcolor{rwthblau100}{\textbf{\gls{nmi}} (ref.)}} & \textcolor{rwthblau100}{\textbf{Sil.}} & \textcolor{rwthblau100}{\textbf{Folds}} \\",
           r"\textcolor{rwthblau100}{} & \textcolor{rwthblau100}{LTs raw} & \textcolor{rwthblau100}{LTs ref.} & \textcolor{rwthblau100}{SR raw} & "
           r"\textcolor{rwthblau100}{SR ref.} & \textcolor{rwthblau100}{LTs} & \textcolor{rwthblau100}{SR} & \textcolor{rwthblau100}{} & \\"]
    blocks = [("All slides with a SpaceRanger reference", None)] + [(f"Slides shared with {n}", n) for n in BASELINE_NAMES]
    rows = []
    for title, bname in blocks:
        have = set(km_m[km_m.ari_sr.notna() & km_m.ari_lts.notna()].held_out_slide)
        if bname is not None and bo is not None and (bo.method == bname).any():
            have &= set(bo[bo.method == bname].held_out_slide)
        elif bname is not None:
            rows.append([r"\multicolumn{9}{l}{\textit{" + title + "}}" + ""] + [""] * 0)
            rows.append([f"{bname}", mt.pending()] + [""] * 7)
            continue
        slides = [s for s in mt.SLIDES if s in have]
        rows.append([r"\multicolumn{9}{l}{\textit{" + title + f" ({len(slides)} slides)" + "}}"])
        for cfg in CFGS_ALL:
            for emb in ("z_trans", "z_token", "z_joint") if cfg == ("nuclei", "svg") else ("z_trans",):
                vals = []
                for col, part in (("ari_lts", "raw"), ("ari_lts", REFINED), ("ari_sr", "raw"), ("ari_sr", REFINED), ("nmi_lts", REFINED), ("nmi_sr", REFINED)):
                    x = ours(km_m, *cfg, emb, part, col)
                    vals.append(mean_or_nan(x[x.index.isin(slides)]))
                sil = ours(km_m, *cfg, emb, REFINED, "silhouette")
                n = len([s for s in slides if s in ours(km_m, *cfg, emb, "raw", "ari_sr").index])
                if not n:
                    continue
                rows.append([f"Ours: {mt.cfg_label(cfg)}, {mt.EMB_LABEL[emb]}"] + [mt.f3(v) for v in vals] + [mt.f3(mean_or_nan(sil[sil.index.isin(slides)])), str(n)])
        if bname is not None and bo is not None:
            vals = []
            for col, kind in (("ari_lts", "pred"), ("ari_lts", "refined_pred"), ("ari_sr", "pred"), ("ari_sr", "refined_pred"), ("nmi_lts", "refined_pred"), ("nmi_sr", "refined_pred")):
                s = baseline_per_slide(bo, bname, kind, col)
                vals.append(mean_or_nan(s[s.index.isin(slides)]))
            rows.append([bname] + [mt.f3(v) for v in vals] + ["--", str(len(slides_with(bo, bname, slides)))])
        rows.append(r"\midrule")
    rows = rows[:-1] if rows and rows[-1] == r"\midrule" else rows
    cap = ("Long summary. \\gls{ari} against the smoothed label-transfer reference (LTs) and SpaceRanger (SR) for the raw and the majority-vote-refined partitions, "
           "\\gls{nmi} for the refined partitions, silhouette of the refined partition in the embedding that was clustered. Ours: mean over Leiden runs with 7--10 "
           "clusters; baselines: mean over $k=7,\\dots,10$ (their own refinement for SpaGCN, none for STAGATE), on the same nuclei. Each block averages only the "
           "slides that the named baseline has, so rows within a block are comparable and blocks are not. The smoothed reference and refined partitions are both "
           "spatially smooth, which inflates their agreement.")
    mt.write(out, "10_tab_long_summary.tex", mt.table("lrrrrrrrr", hdr, rows, "Long summary of all evaluations.", cap,
                                                     "tab:10_long_summary", size=r"\scriptsize", resize=True))


def run(mt_module, out, base):
    global mt
    mt = mt_module
    km = load(base, "final_loso_kmatched_ari.csv")
    if km is None:
        return
    km_m = matched_means(km)
    bo = load(base, "final_loso_baselines_ours_units.csv")
    agree = load(base, "final_loso_partition_agreement.csv")
    dw = load(base, "final_loso_disease_signal_weighted.csv")
    ct = load(base, "final_loso_cluster_annotation_celltypist.csv")
    for step in (lambda: tab_ari_per_sample(km_m, bo, out),
                 lambda: [tab_ari_per_sample(km_m, bo, out, u, v, appendix=True) for u, v in CFGS_MAIN[1:]],
                 lambda: tab_ari_per_sample_embeddings(km_m, out),
                 lambda: tab_baseline_kmatched(km_m, bo, out),
                 lambda: tab_ground_truth(km_m, bo, out),
                 lambda: tab_ari_vs_k(km, out),
                 lambda: tab_partition_agreement(agree, out),
                 lambda: tab_disease_weighted(dw, out),
                 lambda: tab_celltypist(ct, out),
                 lambda: tab_long_summary(km_m, bo, out)):
        try:
            step()
        except Exception as e:  # one broken table must not stop the others
            import traceback
            print(f"  [extended] table failed: {e!r}")
            traceback.print_exc()
