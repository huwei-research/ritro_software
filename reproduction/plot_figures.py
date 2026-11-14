"""Reproduce the numerical figures of RITRO manuscript v3.9 from bundled compact data.

Run from any directory. This script reads saved results only; it never imports
an optimizer or evaluates an objective. No research repository or simulator is required.
"""
from pathlib import Path
import hashlib
import gzip
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "data"
ASSETS = ROOT / "output"
METHODS = ["selected", "int_rfd", "rdse_sb"]
LABELS = ["RITRO", "Int-RFD", "RDSE-SB"]
COLORS = ["#0072B2", "#D55E00", "#009E73"]
STYLES = ["-", "--", "-."]
FAMILIES = ["dictionary", "grassmann", "hyperbolic", "procrustes",
            "rotation", "singular", "spd", "sphere", "torus"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profiles(rows):
    """Retain exact empirical steps; size labels for the manuscript text width."""
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                         "axes.titlesize": 9, "xtick.labelsize": 8.5,
                         "ytick.labelsize": 8.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    evidence = []
    for view in ["tuned", "default"]:
        for kind in ["performance", "data"]:
            fig, axes = plt.subplots(1, 3, figsize=(5.148, 2.15), sharey=True)
            for ax, tol in zip(axes, [1e-3, 1e-5, 1e-7]):
                subset = [r for r in rows if r["view"] == view and r["tau"] == tol]
                assert len(subset) == 810 and not any(r["flat"] for r in subset)
                keys = sorted({(r["problem"], r["seed"]) for r in subset})
                lookup = {(r["problem"], r["seed"], r["method"]): r for r in subset}
                costs = np.array([[lookup[k + (m,)]["first_hit"] or np.inf
                                   for m in METHODS] for k in keys])
                if kind == "performance":
                    best = costs.min(axis=1)
                    values = np.divide(costs, best[:, None], out=np.full_like(costs, np.inf),
                                       where=np.isfinite(best[:, None]))
                    right = 2. ** max(1, int(np.ceil(np.log2(values[np.isfinite(values)].max()))))
                    left = 1.
                    evidence.append(dict(view=view, tolerance=tol,
                                         success=np.isfinite(costs).sum(axis=0).tolist(),
                                         fastest=(values <= 1).sum(axis=0).tolist(),
                                         within_two=(values <= 2).sum(axis=0).tolist()))
                else:
                    dims = np.array([lookup[k + (METHODS[0],)]["dimension"] + 1 for k in keys])
                    values = costs / dims[:, None]
                    left, right = 0., 200.
                for j, (label, color, style) in enumerate(zip(LABELS, COLORS, STYLES)):
                    knots = np.unique(np.r_[left, values[np.isfinite(values[:, j]), j], right])
                    y = np.array([np.mean(values[:, j] <= v) for v in knots])
                    assert np.all(np.diff(y) >= 0)
                    assert np.isclose(y[-1], np.isfinite(costs[:, j]).mean())
                    ax.step(knots, y, where="post", color=color, linestyle=style,
                            linewidth=1.4, label=label)
                if kind == "performance":
                    ax.set_xscale("log", base=2)
                    exp = int(np.log2(right))
                    tick_exp = sorted(set([0, exp] + list(range(0, exp + 1, max(1, int(np.ceil(exp / 4)))))))
                    ticks = 2. ** np.array(tick_exp)
                    ax.set_xticks(ticks, [str(int(v)) for v in ticks])
                    ax.set_xlabel(r"Evaluation ratio $\alpha$")
                else:
                    ax.set_xticks([0, 100, 200])
                    ax.set_xlabel(r"Evaluations / $(d+1)$")
                ax.set_title("("+chr(97+list(axes).index(ax)+(3 if kind=="data" else 0))+") "+r"$\tau=10^{" + str(int(np.log10(tol))) + "}$")
                ax.set_xlim(left, right)
                ax.set_ylim(0, 1.02)
                ax.set_yticks([0, .25, .5, .75, 1])
                ax.grid(axis="y", alpha=.16, linewidth=.45)
            axes[0].set_ylabel("Proportion of instances")
            fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
                       bbox_to_anchor=(.52, 0), ncol=3, frameon=False, fontsize=9)
            fig.subplots_adjust(left=.105, right=.973, top=.87, bottom=.30, wspace=.18)
            for ext in ["pdf", "png"]:
                fig.savefig(ASSETS / f"{view}_{kind}_profiles.{ext}", dpi=200)
            plt.close(fig)
    return evidence


def descents():
    """Reformat the same incumbent quantiles at the manuscript's final size."""
    source = RESULTS / "descent_records.json.gz"
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        records = json.load(stream)["records"]
    names_by_family = {}
    for family in FAMILIES:
        dims = {r["problem"]: r["dimension"] for r in records if r["family"] == family}
        names_by_family[family] = sorted(dims, key=dims.get)
        assert len(dims) == 3
    titles = ["Dictionary learning", "Grassmann estimation", "Hyperbolic fitting",
              "Rectangular Procrustes", "Rotation synchronization", "Singular subspace",
              "SPD mean", "Sphere energy", "Phase synchronization"]
    quantiles = {}
    for index, scale in enumerate(["small", "medium", "large"]):
        fig, axes = plt.subplots(3, 3, figsize=(5.148, 4.90), sharey=True)
        for j, (ax, family, title) in enumerate(zip(axes.flat, FAMILIES, titles)):
            name = names_by_family[family][index]
            group = [r for r in records if r["view"] == "tuned" and r["problem"] == name]
            assert len(group) == 30
            budget, dim = group[0]["budget"], group[0]["dimension"]
            x = np.arange(1, budget + 1)
            for method, label, color, style in zip(METHODS, LABELS, COLORS, STYLES):
                selected = [r for r in group if r["method"] == method]
                assert len(selected) == 10
                curves = [np.asarray(r["relative_error"])[
                    np.searchsorted(r["fe"], x, side="right") - 1] for r in selected]
                q = np.quantile(curves, [.25, .5, .75], axis=0)
                quantiles[f"{name}/{method}"] = hashlib.sha256(q.tobytes()).hexdigest()
                ax.plot(x, np.maximum(q[1], 1e-9), color=color, ls=style, lw=1.3, label=label)
                ax.fill_between(x, np.maximum(q[0], 1e-9), np.maximum(q[2], 1e-9),
                                color=color, alpha=.10)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e-9, 1.5)
            ax.set_yticks([1, 1e-2, 1e-4, 1e-6, 1e-8])
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", alpha=.16, linewidth=.45)
            ax.set_title(f"({chr(97+j)}) {title}\n$d={dim}$", fontsize=8, pad=5)
        fig.supxlabel("Function evaluations", x=.55, y=.065, fontsize=9)
        fig.supylabel("Relative objective error", x=.005, fontsize=9)
        
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center",
                   bbox_to_anchor=(.52, .015), ncol=3, frameon=False, fontsize=9)
        fig.subplots_adjust(left=.11, right=.985, top=.935, bottom=.16, wspace=.16, hspace=.65)
        for ext in ["pdf", "png"]:
            fig.savefig(ASSETS / f"tuned_{scale}_descent.{ext}", dpi=200)
        plt.close(fig)
    return dict(source=str(source.relative_to(ROOT)), sha256=sha256(source),
                quantile_sha256=quantiles, display_floor=1e-9)


def panda_figures():
    """Draw saved simulation outputs at text width, with consistent method colors."""
    source = RESULTS / "panda_results.json"
    bundle = read_json(source)
    rows = bundle["runs"]
    assert len(rows) == 30 and bundle["representative_seed"] == 4201
    methods = ["selected", "int_rfd", "gabo_port"]
    labels = dict(selected=LABELS[0], int_rfd=LABELS[1], gabo_port="GaBO (adapted)")
    colors = dict(selected=COLORS[0], int_rfd=COLORS[1], gabo_port="#009E73")
    styles = dict(selected="-", int_rfd="--", gabo_port="-.")
    quantiles = {}
    fig, axes = plt.subplots(1, 2, figsize=(5.148, 2.40))
    for method in methods:
        group = sorted([r for r in rows if r["method"] == method], key=lambda r: r["seed"])
        assert len(group) == 10
        for ax, field in zip(axes, ["incumbent_cost", "incumbent_late_rms_mm"]):
            values = np.asarray([r[field] for r in group])
            assert values.shape == (10, 300)
            q = np.quantile(values, [.25, .5, .75], axis=0)
            quantiles[f"{method}/{field}"] = hashlib.sha256(q.tobytes()).hexdigest()
            x = np.arange(1, 301)
            ax.plot(x, q[1], color=colors[method], ls=styles[method], lw=1.35, label=labels[method])
            ax.fill_between(x, q[0], q[2], color=colors[method], alpha=.10)
            ax.set_xlabel("Function evaluations")
            ax.set_yscale("log")
            ax.set_xlim(1, 300)
            ax.set_xticks([1, 100, 200, 300])
            ax.grid(axis="y", alpha=.16, linewidth=.45)
    for j,ax in enumerate(axes):
        ax.set_title(f"({chr(97+j)})", fontsize=9, loc="left", pad=6)
    axes[0].set_ylabel("Task cost")
    axes[1].set_ylabel("Late position RMS (mm)")
    axes[1].axhline(20, color="0.5", ls=":", lw=1)
    axes[1].set_yticks([20, 50, 100, 200], ["20", "50", "100", "200"])
    axes[1].minorticks_off()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               bbox_to_anchor=(.52, 0), frameon=False, fontsize=9, columnspacing=1.1)
    fig.subplots_adjust(left=.12, right=.973, top=.87, bottom=.28, wspace=.40)
    for ext in ["pdf", "png"]:
        fig.savefig(ASSETS / f"panda_convergence.{ext}", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(5.148, 2.50))
    target = np.asarray(bundle["task"]["target"])
    for method in methods + ["initial"]:
        trace = bundle["representative_traces"][method]
        position, time = np.asarray(trace["position"]), np.asarray(trace["time"])
        color = colors.get(method, "0.5")
        label = labels.get(method, "Initial design best")
        style = styles.get(method, ":")
        axes[0].plot(position[:, 1], position[:, 2], color=color, ls=style, lw=1.35, label=label)
        axes[1].plot(time, 1000*np.linalg.norm(position-target, axis=1), color=color, ls=style, lw=1.35)
    axes[0].scatter([target[1]], [target[2]], marker="*", s=38, color="black", label="Target", zorder=5)
    starts = np.array([bundle["representative_traces"][m]["position"][0] for m in methods + ["initial"]])
    assert np.allclose(starts, starts[0], rtol=0, atol=1e-12)
    axes[0].scatter([starts[0,1]], [starts[0,2]], marker="o", s=17, facecolors="white", edgecolors="black", label="Start", zorder=6)
    # Direction markers follow existing sampled trajectories; no interpolation of dynamics.
    for method in methods:
        position=np.asarray(bundle["representative_traces"][method]["position"])
        j= max(1, int(len(position)*.08))
        axes[0].annotate("", xy=position[j+1,1:3], xytext=position[j-1,1:3],
            arrowprops=dict(arrowstyle="->", color=colors[method], lw=.8, mutation_scale=7))
    for j,ax in enumerate(axes):
        ax.set_title(f"({chr(97+j)})", fontsize=9, loc="left", pad=6)
    axes[0].set_xlabel("End-effector y (m)")
    axes[0].set_ylabel("End-effector z (m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Position error (mm)")
    axes[1].axhline(20, color="0.5", ls=":", lw=1)
    for ax in axes:
        ax.grid(axis="y", alpha=.16, linewidth=.45)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               bbox_to_anchor=(.52, 0), frameon=False, fontsize=9, columnspacing=1.1)
    fig.subplots_adjust(left=.12, right=.973, top=.87, bottom=.32, wspace=.40)
    for ext in ["pdf", "png"]:
        fig.savefig(ASSETS / f"panda_task_response.{ext}", dpi=200)
    plt.close(fig)
    return dict(source=str(source.relative_to(ROOT)), sha256=sha256(source),
                quantile_sha256=quantiles, representative_seed=4201)




def main():
    import argparse
    import gc
    import tracemalloc
    from unittest.mock import patch
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--check-only", action="store_true", help="Validate data and numerical arrays without saving figures")
    parser.add_argument("--no-tex", action="store_true", help="Use Matplotlib fonts when TeX is unavailable")
    args = parser.parse_args()
    global ASSETS
    ASSETS = args.output.resolve()
    ASSETS.mkdir(parents=True, exist_ok=True)
    manifest = read_json(ROOT / "manifest.json")
    frozen = read_json(ROOT / "expected_figures.json")
    memory = {}
    for name, info in manifest["data"].items():
        path = RESULTS / name
        assert sha256(path) == info["sha256"], f"Modified input: {name}"
        gc.collect()
        tracemalloc.start()
        if name.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
        else:
            payload = read_json(path)
        memory[name] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        del payload
        assert memory[name] <= manifest["budgets"]["reader_python_peak_bytes"]

    rows = read_json(RESULTS / "metrics.json")
    original_plot, original_fill, original_save = Axes.plot, Axes.fill_between, Figure.savefig
    records = []

    def trace_plot(self, *args, **kwargs):
        if len(args) >= 2:
            arrays = [np.ascontiguousarray(np.asarray(a, dtype=float)) for a in args[:2]]
            records.append(["plot", [list(a.shape) for a in arrays], hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest()])
        return original_plot(self, *args, **kwargs)

    def trace_fill(self, x, y1, y2=0, **kwargs):
        arrays = [np.ascontiguousarray(np.asarray(a, dtype=float)) for a in [x, y1, y2]]
        records.append(["fill_between", [list(a.shape) for a in arrays], hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest()])
        return original_fill(self, x, y1, y2, **kwargs)

    def save(self, path, *args, **kwargs):
        if args_cli.check_only:
            return None
        if str(path).endswith(".png") and not args_cli.no_tex:
            import fitz
            with fitz.open(Path(path).with_suffix(".pdf")) as document:
                document[0].get_pixmap(dpi=200).save(path)
            return None
        if str(path).endswith(".pdf"):
            kwargs["metadata"] = {"CreationDate": None, "ModDate": None, "Creator": "RITRO figure rebuild v3.5"}
        return original_save(self, path, *args, **kwargs)

    args_cli = args
    with plt.style.context(str(ROOT / "publication.mplstyle")), patch.object(Axes, "plot", trace_plot), patch.object(Axes, "fill_between", trace_fill), patch.object(Figure, "savefig", save):
        plt.rcParams.update({"font.family": "serif", "font.serif": ["Computer Modern Roman"] if not args.no_tex else ["DejaVu Serif"],
                             "text.usetex": not args.no_tex, "mathtext.fontset": "cm",
                             "legend.fontsize": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                             "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                             "lines.solid_capstyle": "round", "axes.axisbelow": True})
        profile = profiles(rows)
        descent = descents()
        panda = panda_figures()
    assert records == frozen["numeric_artists"], "Plotted data differ from the manuscript"
    assert profile == frozen["profiles"]
    assert descent["quantile_sha256"] == frozen["descents"]["quantile_sha256"]
    assert panda["quantile_sha256"] == frozen["panda"]["quantile_sha256"]
    evidence = {"manuscript": "v3.9", "new_objective_evaluations": 0,
                "numeric_artist_count": len(records), "numeric_arrays_match": True,
                "profiles": profile, "descents": descent, "panda": panda,
                "data_reader_peak_python_bytes": memory,
                "memory_scope": "tracemalloc peak for one JSON input at a time; excludes imported libraries and complete plotting RSS",
                "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
                "tex_text": not args.no_tex,
                "figures_written": not args.check_only}
    (ASSETS / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: evidence[k] for k in ["numeric_artist_count", "numeric_arrays_match", "data_reader_peak_python_bytes", "figures_written"]}, indent=2))


if __name__ == "__main__":
    main()
