import argparse
import itertools
import os
import sys
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


SCENARIO = "impulse"
MODE = "watchall"
WATCH_SEED =0
WATCH_CONTROLLER = "nmpc"
WATCH_NOMINAL = True
CALIB_N = 1


POST_FALL_RUN = 2


ARMS = {
    "nmpc":           dict(label="nmpc", use_ism=False),
    "nmpc_stsmc":     dict(label="nmpc_stsmc", use_ism=True, ism_every_n=1),
    "nmpc_stsmc_200": dict(label="nmpc_stsmc_200", use_ism=True, ism_every_n=2),
    "nmpc_stsmc_80":  dict(label="nmpc_stsmc_80", use_ism=True, ism_every_n=5),
    "nmpc_stsmc_40":  dict(label="nmpc_stsmc_40", use_ism=True, ism_every_n=10),
    "nmpc_dob":       dict(label="nmpc_dob", use_ism=False, use_dob=True, dob_ki_scale=100, dob_dhat_max=50.0),
    "nmpc_stsmc_simplewbc": dict(label="nmpc_stsmc_simplewbc",
                                 use_ism=True, ism_every_n=1,
                                 wbc_full=False),

}

ENABLED = {
    "impulse":     ["nmpc", "nmpc_stsmc", "nmpc_stsmc_200", "nmpc_stsmc_80",
                    "nmpc_stsmc_40", "nmpc_dob", "nmpc_stsmc_simplewbc"],
    "terrain":     ["nmpc", "nmpc_stsmc", "nmpc_stsmc_200", "nmpc_stsmc_80",
                    "nmpc_stsmc_40", "nmpc_dob", "nmpc_stsmc_simplewbc"],
    "degradation": ["nmpc", "nmpc_stsmc", "nmpc_stsmc_200", "nmpc_stsmc_80",
                    "nmpc_stsmc_40", "nmpc_dob", "nmpc_stsmc_simplewbc"],
}

COLORS = {"nmpc": "tab:red", "nmpc_stsmc": "tab:blue",
          "nmpc_stsmc_200": "tab:cyan", "nmpc_stsmc_80": "tab:green",
          "nmpc_stsmc_40": "tab:olive", "nmpc_dob": "tab:orange",
          "nmpc_stsmc_simplewbc": "tab:purple"}


PANELS = [
    (lambda tr: np.abs(tr["roll_err"]),          "|roll err| [rad]",  (0.0, 0.15)),
    (lambda tr: np.abs(tr["pitch_err"]),         "|pitch err| [rad]", (0.0, 0.15)),
    (lambda tr: np.abs(tr["yaw_err"]),           "|yaw err| [rad]",   (0.0, 0.2)),
    (lambda tr: np.abs(tr["vx"] - tr["vx_ref"]), "|vx err| [m/s]",    (0.0, 0.5)),
]


PAPER_LABELS = {"nmpc": "Baseline NMPC",
                "nmpc_stsmc": "NMPC-STSMC (Proposed)"}
PAPER_COLORS = {"nmpc": "tab:blue", "nmpc_stsmc": "tab:orange"}
PAPER_FONT = 15

PAPER_XLIM = {
    "impulse":     None,
    "terrain":     None,
    "degradation": None,
}

PAPER_ERR_YLIM = {
    "impulse":     {"roll": (0, 15), "pitch": (0, 15),
                    "yaw": (0, 15), "vx": (0,1.5)},
    "terrain":     {"roll": (0, 10), "pitch": (0, 10),
                    "yaw": (0, 10), "vx": (0,1)},
    "degradation": {"roll": (0, 12.5), "pitch": (0, 12.5),
                    "yaw": (0, 12.5), "vx": (0,1.25)},
}

PAPER_FORCE_YLIM = {
    "impulse":     {"fx": None, "fy": None, "fz": None},
    "terrain":     {"fx": None, "fy": None, "fz": None},
    "degradation": {"fx": (-50,50), "fy": (-5,5), "fz": None},
}


def _paper_rows(rows):
    by = {r["controller"]: r for r in rows
          if not r.get("error") and r.get("trace") is not None}
    return [(n, by[n]) for n in PAPER_LABELS if n in by]


def plot_paper_errors(scen, rows, tag):
    import matplotlib.pyplot as plt
    sel = _paper_rows(rows)
    if not sel:
        return None
    ylim = PAPER_ERR_YLIM[scen]
    xlim = PAPER_XLIM[scen]
    panels = [("Roll Tracking Error", "Roll Error (deg)", "roll"),
              ("Pitch Tracking Error", "Pitch Error (deg)", "pitch"),
              ("Yaw Tracking Error", "Yaw Error (deg)", "yaw"),
              ("Velocity Tracking Error", "Velocity Error m/s", "vx")]
    fig, axes = plt.subplots(4, 1, figsize=(9, 13), sharex=True)


    for name, row in reversed(sel):
        tr = row["trace"]
        T = tr["T"]
        series = {"roll": np.degrees(np.abs(tr["roll_err"])),
                  "pitch": np.degrees(np.abs(tr["pitch_err"])),
                  "yaw": np.degrees(np.abs(tr["yaw_err"])),
                  "vx": np.abs(tr["vx"] - tr["vx_ref"])}
        z = 3 if name == "nmpc" else 2
        for ax, (_, _, key) in zip(axes, panels):
            ax.plot(T, series[key], color=PAPER_COLORS[name], lw=1.4,
                    zorder=z, label=PAPER_LABELS[name])
    for ax, (title, ylab, key) in zip(axes, panels):
        ax.set_title(title, fontsize=PAPER_FONT + 2, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=PAPER_FONT)
        ax.tick_params(labelsize=PAPER_FONT - 2)
        ax.axhline(0.0, color="gray", lw=0.8, ls="--")
        ax.grid(True, alpha=0.3)
        if ylim.get(key) is not None:
            ax.set_ylim(*ylim[key])
        if xlim is not None:
            ax.set_xlim(*xlim)
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(list(reversed(h)), list(reversed(l)),
                   fontsize=PAPER_FONT - 2)
    axes[-1].set_xlabel("Time (s)", fontsize=PAPER_FONT)
    fig.tight_layout()
    out = os.path.join(PROJECT_ROOT, f"{scen}_paper_errors{tag}.png")
    fig.savefig(out, dpi=300)
    print(f"saved {os.path.relpath(out, PROJECT_ROOT)}")
    return fig


def plot_paper_forces(scen, rows, tag):
    import matplotlib.pyplot as plt
    sel = _paper_rows(rows)
    if not sel:
        return None
    ylim = PAPER_FORCE_YLIM[scen]
    xlim = PAPER_XLIM[scen]
    comp = [("Longitudinal Force $f_x$ (N)", "fx", 0, 5),
            ("Lateral Force $f_y$ (N)", "fy", 1, 6),
            ("Vertical Force $f_z$ (N)", "fz", 2, 7)]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)

    for name, row in reversed(sel):
        tr = row["trace"]
        Tf_, Um, Ui = tr["T_ism"], tr["U_mpc"], tr["U_ism"]
        if len(Tf_) == 0:
            continue
        U = Um + Ui
        z = 3 if name == "nmpc" else 2
        for r, (_, _, iL, iR) in enumerate(comp):
            axes[r, 0].plot(Tf_, U[:, iL], color=PAPER_COLORS[name], lw=1.0,
                            zorder=z, label=PAPER_LABELS[name])
            axes[r, 1].plot(Tf_, U[:, iR], color=PAPER_COLORS[name], lw=1.0,
                            zorder=z)
    axes[0, 0].set_title("Left Leg", fontsize=PAPER_FONT + 2,
                         fontweight="bold")
    axes[0, 1].set_title("Right Leg", fontsize=PAPER_FONT + 2,
                         fontweight="bold")
    for r, (ylab, key, _, _) in enumerate(comp):
        for c in (0, 1):
            ax = axes[r, c]
            ax.tick_params(labelsize=PAPER_FONT - 2)
            ax.axhline(0.0, color="gray", lw=0.8, ls="--")
            ax.grid(True, alpha=0.3)
            if ylim.get(key) is not None:
                ax.set_ylim(*ylim[key])
            if xlim is not None:
                ax.set_xlim(*xlim)
        axes[r, 0].set_ylabel(ylab, fontsize=PAPER_FONT)
    for c in (0, 1):
        axes[-1, c].set_xlabel("Time (s)", fontsize=PAPER_FONT)
    fig.suptitle("Control Force Components", fontsize=PAPER_FONT + 3,
                 fontweight="bold")
    h, l = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(list(reversed(h)), list(reversed(l)),
                      fontsize=PAPER_FONT - 3.5, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(PROJECT_ROOT, f"{scen}_paper_forces{tag}.png")
    fig.savefig(out, dpi=300)
    print(f"saved {os.path.relpath(out, PROJECT_ROOT)}")
    return fig


TERRAIN_PLOTS = False
TERRAIN_PLOT_DIR = "terrain_plots"


def _worker_init():
    sys.path.insert(0, PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)


def _sim():
    import controller
    return controller


_CMP_IMP = [
    ("peak_pitch_err", "pitch", 3, 6, 1.0),
    ("peak_roll_err", "roll", 3, 6, 1.0),
    ("peak_yaw_err", "yaw", 3, 6, 1.0),
    ("rms_vx_err", "rms_vx", 4, 7, 1.0),
    ("recovery_att", "recov", 2, 5, 1.0),
]
_CMP_DEG = [
    ("ss_pitch_err", "ss_pitch", 2, 9, np.degrees(1.0)),
    ("ss_vx_err", "ss_vx", 3, 8, 1.0),
    ("ss_roll_err", "ss_roll", 2, 8, np.degrees(1.0)),
    ("ss_yaw_err", "ss_yaw", 2, 8, np.degrees(1.0)),
    ("rms_vx_err", "rms_vx", 3, 7, 1.0),
    ("ss_theta_eq", "theta_eq", 2, 8, np.degrees(1.0)),
]
_CMP_TER = [
    ("peak_roll_err", "pk_roll", 3, 7, 1.0),
    ("peak_pitch_err", "pk_pitch", 3, 8, 1.0),
    ("rms_roll_err", "rms_roll", 3, 8, 1.0),
    ("rms_pitch_err", "rms_pit", 3, 7, 1.0),
    ("rms_yaw_err", "rms_yaw", 3, 7, 1.0),
    ("rms_vx_err", "rms_vx", 3, 7, 1.0),
]

PROFILES = {
    "impulse": dict(seed_base=0, out="mc_results.csv", full_n=250,
                    procs=3, tally_every=1, cmp_cols=_CMP_IMP,
                    signed_best=False),
    "terrain": dict(seed_base=0, out="mc_terrain.csv", full_n=250,
                    procs=5, tally_every=10, cmp_cols=_CMP_TER,
                    signed_best=False),
    "degradation": dict(seed_base=0, out="deg_results.csv", full_n=100,
                        procs=3, tally_every=2, cmp_cols=_CMP_DEG,
                        signed_best=True),
}


def _draw(scen, seed):
    sim = _sim()
    return {"impulse": sim.draw_trial_params,
            "terrain": sim.draw_terrain_params,
            "degradation": sim.draw_degradation_params}[scen](seed)


def _nominal(scen):
    sim = _sim()
    return {"impulse": sim.nominal_trial_params,
            "terrain": sim.nominal_terrain_params,
            "degradation": sim.nominal_degradation_params}[scen]()


def _run_one(args):
    scen, seed, name, nominal, keep_trace = args
    sim = _sim()
    tp = _nominal(scen) if nominal else _draw(scen, seed)
    row = sim.run_trial(seed, sim.ControllerCfg(**ARMS[name]), tp=tp,
                        keep_trace=keep_trace)
    row["seed"] = seed
    return row


_COLOR = bool(os.environ.get("FORCE_COLOR")) or sys.stdout.isatty()


def _c(s, code):
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


def _num(v):
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _f(row, key, nd=3, w=6, scale=1.0):
    v = _num(row.get(key))
    return f"{v * scale:{w}.{nd}f}" if v is not None else "-".rjust(w)


def _result_tag(row):
    if row.get("error"):
        return _c("ERR ", "35")
    if row.get("fall"):
        return _c("FALL", "31")
    return _c(" ok ", "32")


def _fmt_progress(scen, row, k, total, t_start):
    el = time.time() - t_start
    eta = (el / max(k, 1)) * (total - k)
    ftime = _num(row.get("fall_time"))
    ft = f" @{ftime:5.2f}s" if row.get("fall") and ftime is not None else " " * 7
    head = (f"[{k:5d}/{total}] seed={row.get('seed', -1):5d} "
            f"{str(row.get('controller', '?')):16s} {_result_tag(row)}{ft} | ")
    if scen == "impulse":
        mid = (f"pitch={_f(row, 'peak_pitch_err')} "
               f"roll={_f(row, 'peak_roll_err')} "
               f"vx={_f(row, 'rms_vx_err', 4, 7)} "
               f"rec={_f(row, 'recovery_att', 2, 5)} | ")
    elif scen == "degradation":
        mid = (f"ss_pitch={_f(row, 'ss_pitch_err', 2, 6, np.degrees(1.0))}deg "
               f"theta_eq={_f(row, 'ss_theta_eq', 2, 6, np.degrees(1.0))}deg "
               f"ss_vx={_f(row, 'ss_vx_err', 3, 7)} "
               f"ss_roll={_f(row, 'ss_roll_err', 2, 6, np.degrees(1.0))}deg | ")
    else:
        tag = ("ERR " if row.get("error")
               else "FALL" if row.get("fall") else " ok ")
        return (f"[{k:3d}/{total}] seed={row.get('seed', -1):4d} "
                f"{str(row.get('controller', '?')):12s} {tag} | "
                f"rms_roll={row.get('rms_roll_err', float('nan')):.3f} "
                f"rms_pitch={row.get('rms_pitch_err', float('nan')):.3f} "
                f"rms_vx={row.get('rms_vx_err', float('nan')):.3f} | "
                f"el {el/60:4.1f}m eta {eta/60:5.1f}m")
    return (head + mid
            + f"cone={str(row.get('cone_violations', '-')):>3s} "
              f"sat={str(row.get('sat_count', '-')):>5s} "
              f"mpcfail={str(row.get('mpc_fail_count', '-')):>3s} "
              f"{_f(row, 'mean_solve_ms', 1, 5)}ms | "
              f"el {el / 60:5.1f}m eta {eta / 60:6.1f}m")


def _scen_line(scen, tp_row):
    if scen == "impulse":
        return (f"imp {_num(tp_row.get('tp_imp_mag')) or 0:.0f}N "
                f"@{_num(tp_row.get('tp_imp_time')) or 0:.1f}s "
                f"dir {np.degrees(_num(tp_row.get('tp_imp_dir')) or 0):.0f}deg, "
                f"mu {_num(tp_row.get('tp_mu_ground')) or 0:.2f}")
    if scen == "degradation":
        return (f"gain_min {_num(tp_row.get('tp_gain_min')) or 1.0:.2f}, "
                f"friction {_num(tp_row.get('tp_joint_friction')) or 0:.2f} Nm, "
                f"mu {_num(tp_row.get('tp_mu_ground')) or 0:.2f}")
    return (f"bump {_num(tp_row.get('tp_terrain_bump_height')) or 0:.2f} m, "
            f"mu scale {_num(tp_row.get('tp_terrain_mu_scale')) or 0:.2f}")


def _seed_compare(scen, seed_rows, order):
    prof = PROFILES[scen]
    cmp_cols = prof["cmp_cols"]
    by = {str(r.get("controller")): r for r in seed_rows}
    names = [n for n in order if n in by] + [n for n in by if n not in order]
    seed = seed_rows[0].get("seed")
    scen_desc = _scen_line(scen, seed_rows[0])

    best = {}
    for key, _, _, _, _ in cmp_cols:
        if scen == "degradation" and key == "ss_theta_eq":
            continue
        cand = []
        for n in names:
            r = by[n]
            if r.get("fall") or r.get("error"):
                continue
            v = _num(r.get(key))
            if v is not None:
                cand.append((abs(v) if prof["signed_best"] else v, n))
        if cand:
            best[key] = min(cand)[1]

    w = max(len(n) for n in names)
    head = (f"  | {'controller':{w}s}  res  "
            + "  ".join(f"{h:>{wd}s}" for _, h, _, wd, _ in cmp_cols)
            + "   cone    sat")
    lines = [_c(f"  +-- seed {seed}  ({scen_desc})", "36"), _c(head, "2")]
    for n in names:
        r = by[n]
        cells = []
        for key, _, nd, wd, sc in cmp_cols:
            v = _num(r.get(key))
            s = f"{v * sc:.{nd}f}" if v is not None else "-"
            if best.get(key) == n:
                cells.append(_c(f"*{s}", "1;32").rjust(wd + (9 if _COLOR else 0)))
            else:
                cells.append(s.rjust(wd))
        lines.append(f"  | {n:{w}s} {_result_tag(r)} "
                     + "  ".join(cells)
                     + f"  {str(r.get('cone_violations', '-')):>5s}"
                       f"  {str(r.get('sat_count', '-')):>5s}")
        if r.get("error"):
            lines.append(_c(f"  |   {n}: {r['error']}", "35"))
    return "\n".join(lines)


def _rolling_tally(scen, rows):
    df = pd.DataFrame(rows)
    if "controller" not in df.columns:
        return ""
    cmp_cols = PROFILES[scen]["cmp_cols"]
    out = [_c(f"  ==== after {len(df)} trials ====", "36")]
    for name, g in df.groupby("controller"):
        n = len(g)
        falls = int(g["fall"].sum()) if "fall" in g else 0
        ok = g[~g["fall"].astype(bool)] if "fall" in g else g
        bits = [f"  {str(name):18s} n={n:5d} falls={falls:4d} "
                f"({100.0 * falls / max(n, 1):4.1f}%) n_ok={len(ok):4d}"]
        for key, hdr, nd, _, sc in cmp_cols:
            if key not in g:
                continue
            v = ok[key].dropna()
            if len(v):
                bits.append(f"{hdr}={v.mean() * sc:.{nd}f}")
        if "error" in g:
            ne = int(g["error"].notna().sum())
            if ne:
                bits.append(_c(f"errs={ne}", "35"))
        out.append(" | ".join(bits))
    return "\n".join(out)


def _ordered_names(scen, df):
    order = ENABLED[scen]
    return ([n for n in order if n in df["controller"].unique()]
            + [n for n in df["controller"].unique() if n not in order])


def summarize_impulse(df):
    if "error" in df.columns:
        df = df[df["error"].isna()]
    names = _ordered_names("impulse", df)
    g = {n: df[df["controller"] == n].set_index("seed") for n in names}

    surv_sets = [set(g[n].index[~g[n]["fall"].astype(bool)]) for n in names]
    common = sorted(set.intersection(*surv_sets)) if surv_sets else []

    metrics = ["peak_roll_err", "peak_pitch_err", "peak_yaw_err",
               "rms_vx_err", "cone_violations"]
    print(f"\n===== SUMMARY (mean ± std over ALL trials, (ok: non-fall "
          f"only); recovery on the {len(common)} common-survivor seeds) "
          f"=====")
    for n in names:
        d = g[n]
        N = len(d)
        falls = int(d["fall"].sum())
        ok = d[~d["fall"].astype(bool)]
        line = (f"{n:18s} n={N:4d} falls={falls}/{N} "
                f"({100 * falls / max(N, 1):.0f}%) n_ok={len(ok):4d}")
        for m in metrics:
            if m not in d:
                continue
            v_all = d[m].dropna()
            v_ok = ok[m].dropna()
            if len(v_all):
                line += f" | {m}={v_all.mean():.3f}±{v_all.std():.3f}"
                if len(v_ok):
                    line += f" (ok:{v_ok.mean():.3f}±{v_ok.std():.3f})"
        if len(ok):
            rec_n = int(ok["recovery_att"].notna().sum())
            line += f" | recovered={rec_n}/{len(ok)} survivors"
        r = d.loc[d.index.isin(common), "recovery_att"].dropna()
        if len(r):
            line += f" | recovery={r.mean():.2f}±{r.std():.2f}"
        print(line)

    print("\n===== PAIRWISE COMMON-SURVIVOR RECOVERY (all pairs) =====")
    for a, b in itertools.combinations(names, 2):
        da, db = g[a], g[b]
        shared = da.index.intersection(db.index)
        both = [s for s in shared
                if not da.loc[s, "fall"] and not db.loc[s, "fall"]]
        if len(both) < 3:
            continue
        ra = da.loc[both, "recovery_att"].dropna()
        rb = db.loc[both, "recovery_att"].dropna()
        if not len(ra) or not len(rb):
            continue
        print(f"{a:18s} vs {b:18s} on {len(both):3d} shared survivors: "
              f"{ra.mean():.2f}±{ra.std():.2f} s vs "
              f"{rb.mean():.2f}±{rb.std():.2f} s  "
              f"(delta {rb.mean() - ra.mean():+.2f})")


def summarize_degradation(df):
    if "error" in df.columns:
        df = df[df["error"].isna()]
    names = _ordered_names("degradation", df)
    g = {n: df[df["controller"] == n].set_index("seed") for n in names}

    print(f"\n===== SUMMARY (mean ± std over ALL trials, (ok: non-fall only)) "
          f"=====")
    print("signed means: the claim is that the fast layer drives these to "
          "zero, so sign and magnitude both matter")
    metrics = [("ss_pitch_err", np.degrees(1.0), "deg"),
               ("ss_theta_eq", np.degrees(1.0), "deg"),
               ("ss_vx_err", 1.0, "m/s"),
               ("ss_roll_err", np.degrees(1.0), "deg"),
               ("ss_yaw_err", np.degrees(1.0), "deg"),
               ("rms_vx_err", 1.0, "m/s"),
               ("cone_violations", 1.0, "")]
    for n in names:
        d = g[n]
        N = len(d)
        falls = int(d["fall"].sum())
        ok = d[~d["fall"].astype(bool)]
        line = (f"{n:18s} n={N:4d} falls={falls}/{N} "
                f"({100 * falls / max(N, 1):.0f}%) n_ok={len(ok):4d}")
        for m, sc, unit in metrics:
            if m not in d:
                continue
            v_all = d[m].dropna()
            v_ok = ok[m].dropna()
            if len(v_all):
                line += (f" | {m}={v_all.mean() * sc:.3f}"
                         f"±{v_all.std() * sc:.3f}")
                if len(v_ok):
                    line += (f" (ok:{v_ok.mean() * sc:.3f}"
                             f"±{v_ok.std() * sc:.3f})")
                if unit:
                    line += f" {unit}"
        print(line)

    print("\n===== PAIRWISE COMMON-SURVIVOR STEADY STATE (all pairs) =====")
    for a, b in itertools.combinations(names, 2):
        da, db = g[a], g[b]
        shared = da.index.intersection(db.index)
        both = [s for s in shared
                if not da.loc[s, "fall"] and not db.loc[s, "fall"]]
        if len(both) < 3:
            continue
        pa = da.loc[both, "ss_pitch_err"].dropna() * np.degrees(1.0)
        pb = db.loc[both, "ss_pitch_err"].dropna() * np.degrees(1.0)
        va = da.loc[both, "ss_vx_err"].dropna()
        vb = db.loc[both, "ss_vx_err"].dropna()
        if not len(pa) or not len(pb):
            continue
        print(f"{a:18s} vs {b:18s} on {len(both):3d} shared survivors: "
              f"pitch {pa.mean():+.2f} vs {pb.mean():+.2f} deg  "
              f"(delta {pb.mean() - pa.mean():+.2f})  |  "
              f"vx {va.mean():+.3f} vs {vb.mean():+.3f} m/s  "
              f"(delta {vb.mean() - va.mean():+.3f})")


def summarize_terrain(df):
    print("\n===== TERRAIN SUMMARY (fall rate: all trials; "
          "tracking stats: mean ± std over NON-FALL trials) ====")
    for name in _ordered_names("terrain", df):
        d = df[df["controller"] == name]
        if not len(d):
            continue
        falls = int(d["fall"].astype(bool).sum())
        ok = d[~d["fall"].astype(bool)]
        line = (f"{name:12s} n={len(d):3d} "
                f"falls={falls}/{len(d)} ({100*falls/len(d):.0f}%) "
                f"n_ok={len(ok):3d}")
        for m in ["peak_roll_err", "peak_pitch_err", "peak_yaw_err",
                  "rms_roll_err", "rms_pitch_err", "rms_yaw_err",
                  "rms_vx_err", "peak_vx_err"]:
            if m not in ok:
                continue
            v = ok[m].dropna()
            if len(v):
                line += f" | {m}={v.mean():.3f}±{v.std():.3f}"
        print(line)


SUMMARIZE = {"impulse": summarize_impulse, "terrain": summarize_terrain,
             "degradation": summarize_degradation}


def _plot_rows(plt, scen, seed, rows, title_extra=""):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for row in rows:
        name = row["controller"]
        c = COLORS.get(name, "k")
        tr = row.get("trace")
        if tr is None:
            continue
        T = tr["T"]
        fell = bool(row["fall"])
        lbl = name + (" (FELL)" if fell else "")
        for ax, (fn, _, _) in zip(axes, PANELS):
            ax.plot(T, fn(tr), color=c, lw=1.1,
                    label=lbl if ax is axes[0] else None)
        if fell and np.isfinite(row.get("fall_time", np.nan)):
            for ax in axes:
                ax.axvline(row["fall_time"], color=c, lw=1.0, ls=":",
                           alpha=0.8)
    for ax, (_, ylab, ylim) in zip(axes, PANELS):
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f"{scen} seed {seed}  {title_extra}".rstrip())
    axes[0].legend(fontsize=9)
    axes[-1].set_xlabel("t [s]")
    fig.tight_layout()
    return fig


def _plot_terrain_seed(seed, rows, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tp = _draw("terrain", seed)
    fig = _plot_rows(plt, "terrain", seed, rows,
                     f"(bump amp {tp.terrain_bump_height:.2f} m)")
    out = os.path.join(plot_dir, f"terrain_seed{seed:03d}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_nominal(scen, by_name, tag):
    if not by_name:
        return
    sim = _sim()
    if scen == "degradation":
        sim.plot_all(by_name, sim.SS_START, tag=tag, save=True, show=True)
        return
    import matplotlib.pyplot as plt
    rows = list(by_name.values())
    fig = _plot_rows(plt, scen, rows[0].get("seed"), rows, "(nominal)")
    fname = os.path.join(PROJECT_ROOT, f"{scen}_nominal{tag}.png")
    fig.savefig(fname, dpi=150)
    print(f"saved {os.path.relpath(fname, PROJECT_ROOT)}")
    plt.show()


def _abort_if_broken(row):
    if row.get("error"):
        print("\nFIRST TRIAL FAILED — aborting campaign.\n")
        print("error:", row["error"])
        print(row.get("traceback", "(no traceback captured)"))
        sys.exit(1)


def _validation_line(scen, first):
    if scen == "impulse":
        return ("validation trial OK: "
                f"fall={first['fall']} (t={first.get('fall_time')}), "
                f"impulse at t={first.get('tp_imp_time'):.1f}s "
                f"{first.get('tp_imp_mag'):.0f}N, "
                f"mu={first.get('tp_mu_ground'):.2f}, "
                f"push point=({first.get('imp_point_x'):.3f}, "
                f"{first.get('imp_point_y'):.3f}, "
                f"{first.get('imp_point_z'):.3f}), "
                f"mpc_fails={first.get('mpc_fail_count')}, "
                f"solve={first['mean_solve_ms']:.1f}ms")
    if scen == "degradation":
        return ("validation trial OK: "
                f"fall={first['fall']} (t={first.get('fall_time')}), "
                f"gain_min={first.get('tp_gain_min'):.2f}, "
                f"friction={first.get('tp_joint_friction'):.2f} Nm, "
                f"mu={first.get('tp_mu_ground'):.2f}, "
                f"mpc_fails={first.get('mpc_fail_count')}, "
                f"solve={first['mean_solve_ms']:.1f}ms")
    return ("validation trial OK: "
            f"fall={first['fall']} "
            f"rms_roll={first.get('rms_roll_err', float('nan')):.3f} "
            f"rms_vx={first.get('rms_vx_err', float('nan')):.3f}")


def _strip(row):
    return {k: v for k, v in row.items()
            if not isinstance(v, (dict, list, np.ndarray))}


def do_watch(scen, seed, name, no_plot, tag):
    _worker_init()
    sim = _sim()
    if scen == "terrain":


        tp = _draw("terrain", seed)
        names = ENABLED["terrain"] if name == "both" else [name]
        rows = []
        for nm in names:
            print(f"\nterrain watch: seed {seed}, {nm}, "
                  f"bump amp {tp.terrain_bump_height:.2f} m")
            row = sim.run_trial(seed, sim.ControllerCfg(**ARMS[nm]), tp=tp,
                                gui=True, keep_trace=True)
            for k, v in row.items():
                if k not in ("trace", "traceback"):
                    print(f"{k}: {v}")
            if row.get("error"):
                print(row.get("traceback", ""))
                continue
            rows.append(row)
        if not any("trace" in r for r in rows):
            return
        import matplotlib.pyplot as plt
        _plot_rows(plt, "terrain", seed, rows,
                   f"(bump amp {tp.terrain_bump_height:.2f} m)")
        plt.show()
        return

    nominal = WATCH_NOMINAL
    tp = _nominal(scen) if nominal else _draw(scen, seed)
    tp.post_fall_run = POST_FALL_RUN
    if scen == "degradation":
        print(sim.describe(tp), flush=True)
    row = sim.run_trial(seed, sim.ControllerCfg(**ARMS[name]), tp=tp,
                        gui=True, keep_trace=True)
    _abort_if_broken(row)
    for k, v in row.items():
        if k != "trace":
            print(f"{k}: {v}")
    if scen == "degradation" and not no_plot and not row.get("error"):
        _plot_nominal(scen, {name: row}, tag)


def do_watchall(scen, seed, no_plot, tag):
    _worker_init()
    sim = _sim()
    names = ENABLED[scen]
    thing = {"impulse": "shove", "terrain": "terrain",
             "degradation": "degradation"}[scen]

    tp = _nominal(scen)
    tp.post_fall_run = POST_FALL_RUN
    seed = tp.seed
    if scen == "degradation":
        print(sim.describe(tp), flush=True)
    print(f"Playing the nominal {scen} scenario with every controller — "
          f"the SAME {thing}, floor, and noise each time. Next trial "
          "starts automatically.")
    t_start = time.time()
    rows = []
    for k, name in enumerate(names, 1):
        print(f"\n=== seed {seed} | {name} ({k}/{len(names)}) ===",
              flush=True)
        row = sim.run_trial(seed, sim.ControllerCfg(**ARMS[name]), tp=tp,
                            gui=True, keep_trace=True)
        row["seed"] = seed
        rows.append(row)
        print(_fmt_progress(scen, row, k, len(names), t_start), flush=True)
        if row.get("error"):

            print("ERROR:", row["error"])
            print(row.get("traceback", ""))
    print()
    print(_seed_compare(scen, rows, names), flush=True)
    if not no_plot:
        ok_rows = [r for r in rows if not r.get("error")]
        if not ok_rows:
            return
        import matplotlib.pyplot as plt


        plot_paper_errors(scen, ok_rows, tag)
        plot_paper_forces(scen, ok_rows, tag)
        if scen == "degradation":
            _plot_nominal(scen, {r["controller"]: r for r in ok_rows}, tag)
        else:
            _plot_rows(plt, scen, seed, ok_rows)
            plt.show()


def do_determinism(scen):
    _worker_init()
    r1 = _run_one((scen, 0, "nmpc_stsmc", False, False))
    _abort_if_broken(r1)
    r2 = _run_one((scen, 0, "nmpc_stsmc", False, False))
    keys = [k for k in r1
            if isinstance(r1[k], float) and not np.isnan(r1[k])
            and k != "mean_solve_ms"]
    diffs = {k: (r1[k], r2[k]) for k in keys if not np.isclose(r1[k], r2[k])}
    print("DETERMINISTIC" if not diffs else f"NON-DETERMINISTIC: {diffs}")


def do_nominal(scen, controllers, out_path, plotting, tag):
    _worker_init()
    sim = _sim()
    tp = _nominal(scen)
    if scen == "degradation":
        print(sim.describe(tp), flush=True)
    rows, by_name = [], {}
    t_start = time.time()
    for k, name in enumerate(controllers, 1):
        row = _run_one((scen, tp.seed, name, True, plotting))
        if k == 1:
            _abort_if_broken(row)
        rows.append(row)
        if not row.get("error"):
            by_name[name] = row
        print(_fmt_progress(scen, row, k, len(controllers), t_start),
              flush=True)
    print()
    print(_seed_compare(scen, rows, controllers), flush=True)
    df = pd.DataFrame([_strip(r) for r in rows])
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(df)} rows)")
    if plotting:
        _plot_nominal(scen, by_name, tag)


def do_campaign(scen, args, out_path):
    prof = PROFILES[scen]
    _worker_init()
    seeds = (args.seeds if args.seeds
             else list(range(prof["seed_base"], prof["seed_base"] + args.n)))
    controllers = args.controllers
    keep_trace = scen == "terrain" and not args.no_plot and TERRAIN_PLOTS
    jobs = [(scen, s, c, False, keep_trace) for s in seeds
            for c in controllers]
    total = len(jobs)
    plot_dir = os.path.join(PROJECT_ROOT, TERRAIN_PLOT_DIR)
    if keep_trace:
        os.makedirs(plot_dir, exist_ok=True)
    if len(seeds) >= 50:
        print(_c("PREFLIGHT (full campaign): determinism checked? "
                 "calib fall-rates in the discriminating band? "
                 "draw params docstring synced? fresh --out file?",
                 "33"), flush=True)

    print(f"{scen} campaign: {len(seeds)} seeds x {len(controllers)} "
          f"controllers = {total} trials, {args.procs} proc(s)")
    print(f"validation trial: seed={jobs[0][1]} controller={jobs[0][2]} ...")
    first = _run_one(jobs[0])
    _abort_if_broken(first)
    print(_validation_line(scen, first))
    print(flush=True)

    rows = [first]
    remaining = jobs[1:]
    t_start = time.time()
    by_seed = {first["seed"]: [first]}
    count = [1]

    def _save():
        pd.DataFrame([_strip(r) for r in rows]).to_csv(out_path, index=False)

    def _tripwire():
        errs = [r for r in rows if r.get("error")]
        if len(errs) >= 5 and len(errs) >= len(rows) - 1:
            _save()
            print("\nTRIALS ARE MASS-FAILING — aborting campaign.\n")
            print("error:", errs[0]["error"])
            print(errs[0].get("traceback", "(no traceback captured)"))
            sys.exit(1)

    def _ingest(row):
        if row is not first:
            rows.append(row)
        count[0] += 0 if row is first else 1
        k = count[0]
        _tripwire()
        if scen == "terrain":
            print(_fmt_progress(scen, row, k, total, t_start), flush=True)
            s = row["seed"]
            by_seed.setdefault(s, [])
            if row is not first:
                by_seed[s].append(row)
            if len(by_seed[s]) == len(controllers):
                if keep_trace:
                    path = _plot_terrain_seed(s, by_seed[s], plot_dir)
                    print(f"      saved "
                          f"{os.path.relpath(path, PROJECT_ROOT)}",
                          flush=True)
                for r in by_seed.pop(s):
                    r.pop("trace", None)
        if k % prof["tally_every"] == 0:
            _save()
            if scen != "terrain":
                print(_rolling_tally(scen, [_strip(r) for r in rows]),
                      flush=True)

    _ingest(first)
    if args.procs <= 1 or args.smoke:
        for j in remaining:
            _ingest(_run_one(j))
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.procs, initializer=_worker_init) as pool:
            for row in pool.imap(_run_one, remaining):
                _ingest(row)

    df = pd.DataFrame([_strip(r) for r in rows])
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(df)} rows)")
    if "error" in df.columns:
        errs = df[df["error"].notna()]
        if len(errs):
            print(f"WARNING: {len(errs)} trials errored; "
                  f"first: {errs.iloc[0]['error']}")
            if "traceback" in errs.columns:
                print(errs.iloc[0].get("traceback", ""))
    SUMMARIZE[scen](df)
    return df


def main():
    ap = argparse.ArgumentParser(description="unified 3-scenario runner")
    ap.add_argument("--scenario", default=SCENARIO,
                    choices=list(PROFILES))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="run these specific seeds instead of the range")
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--controllers", nargs="+", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--nominal", action="store_true",
                    help="the paper scenario, one run per arm, serial, plots")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check-determinism", action="store_true")
    ap.add_argument("--watch", nargs=2, metavar=("SEED", "CONTROLLER"),
                    help="one trial with GUI; terrain accepts 'both' to "
                         "run every arm overlaid (sep32 style)")
    ap.add_argument("--watchall", nargs=1, metavar="SEED",
                    help="play one seed's trial with every controller, GUI")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for figure filenames")

    if len(sys.argv) == 1:
        print(f"no arguments given -> SCENARIO='{SCENARIO}' MODE='{MODE}'")
        mode_args = {
            "watch": ["--watch", str(WATCH_SEED), WATCH_CONTROLLER],
            "watchall": ["--watchall", str(WATCH_SEED)],
            "smoke": ["--smoke"],
            "determinism": ["--check-determinism"],
            "nominal": ["--nominal"],
            "calib": ["--n", str(CALIB_N)],
            "full": [],
        }[MODE]
        args = ap.parse_args(["--scenario", SCENARIO] + mode_args)
    else:
        args = ap.parse_args()

    scen = args.scenario
    prof = PROFILES[scen]


    for s in ENABLED:
        ENABLED[s] = [n for n in ENABLED[s] if n in ARMS]
    if args.controllers is not None:
        missing = [n for n in args.controllers if n not in ARMS]
        if missing:
            print(f"skipping unknown/disabled arms: {missing}")
        args.controllers = [n for n in args.controllers if n in ARMS]
    if args.n is None:
        args.n = prof["full_n"]
    if args.procs is None:
        args.procs = prof["procs"]
    if args.controllers is None:
        args.controllers = list(ENABLED[scen])
    out_path = os.path.join(PROJECT_ROOT, args.out or prof["out"])
    tag = f"_{args.tag}" if args.tag else ""

    if args.watch:
        do_watch(scen, int(args.watch[0]), args.watch[1], args.no_plot, tag)
        return
    if args.watchall:
        do_watchall(scen, int(args.watchall[0]), args.no_plot, tag)
        return
    if args.check_determinism:
        do_determinism(scen)
        return
    if args.nominal:
        do_nominal(scen, args.controllers, out_path,
                   not args.no_plot, tag)
        return

    if args.smoke:
        args.n = 2
        args.procs = 1
    do_campaign(scen, args, out_path)


    if scen == "degradation" and not args.no_plot and not args.smoke:
        print("\nplotting the nominal scenario for the figures "
              "(campaign rows do not carry traces)")
        by_name = {}
        for name in args.controllers:
            row = _run_one((scen, _nominal(scen).seed, name, True, True))
            if not row.get("error"):
                by_name[name] = row
        _plot_nominal(scen, by_name, tag)


if __name__ == "__main__":
    main()
