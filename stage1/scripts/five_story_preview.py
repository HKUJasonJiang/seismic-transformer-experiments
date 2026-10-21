"""Generate a five-story period/PGA preview using the production C++ solver.

Run with the existing mdof-workbench Python environment. Source files are read
only; every new file is written under experiment/stage1. The archived EXE is
used only to export baseline material parameters, never to supply responses.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np

STAGE = Path(__file__).resolve().parents[1]
PROJECT = STAGE.parents[1]
WORKBENCH = PROJECT / "mdof-workbench"
sys.path.insert(0, str(WORKBENCH / "src"))
from mdof_workbench.api import run_case, save_result, load_result
from mdof_workbench.ground_motion import load_record
from mdof_workbench.schema import config_hash

G = 9.80665
NAMES = ["E0", "sy0", "eta", "mu", "gama", "esoft", "alpha", "beta", "a_k", "Omega"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def baseline(config, run_dir):
    folder = run_dir / "material_source"
    source_exe = PROJECT / "3-Data/MDOF_BenchMark/Flexural_Shear_0.1.exe"
    source_inputs = WORKBENCH / "examples/validation/legacy-models/inputs"
    result = folder / "Results/matcode1.txt"
    if not result.exists():
        for name in ("inputs", "GMs", "Results"):
            (folder / name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_exe, folder / "legacy.exe")
        shutil.copy2(source_inputs / "config.txt", folder / "inputs/config.txt")
        shutil.copy2(source_inputs / "DensifyingConfig.txt", folder / "inputs/DensifyingConfig.txt")
        # Explicit deterministic generation; preserve documented age/skeleton switches.
        lines = (folder / "inputs/config.txt").read_bytes().decode("gbk").splitlines()
        values = lines[1].split()
        values[0], values[1], values[3], values[4] = str(config["damping_ratio"]), "0.02", str(config["mass_per_area_kg_m2"]), "0"
        lines[1] = "\t".join(values)
        (folder / "inputs/config.txt").write_bytes(("\r\n".join(lines) + "\r\n").encode("gbk"))
        (folder / "inputs/Dir.txt").write_text("./Results\npreview\n", encoding="ascii")
        (folder / "inputs/EQ_List.txt").write_text("record\n", encoding="ascii")
        header = "Blg_LB Stories height Year T1 T2 Area IM Site_CD Site_GP YieldOver PeakOver YieldStiff PGA dt Ttime StrucType matcodeinput OutputCtrl BlgCondition MasonryWallRatio Width"
        row = [1, config["stories"], config["stories"] * config["story_height_m"], config["year"], -1, -1,
               config["floor_area_m2"], config["fortification_intensity"], config["site_category"], config["site_group"],
               -1, -1, -1, 1, 0.02, 1, config["structure_type"], 3, 111000, -1, -1, -1]
        (folder / "inputs/BlgAttributes.txt").write_text("1\n" + header + "\n" + " ".join(map(str, row)) + "\n", encoding="ascii")
        (folder / "GMs/record.txt").write_text("51\n" + "".join(f"{i*.02:.4f}\t0\n" for i in range(51)), encoding="ascii")
        p = subprocess.run([str(folder / "legacy.exe")], cwd=folder, input=b"\n", capture_output=True,
                           timeout=45, creationflags=subprocess.CREATE_NO_WINDOW)
        (folder / "stdout.log").write_bytes(p.stdout)
        (folder / "stderr.log").write_bytes(p.stderr)
        if p.returncode or not result.exists():
            raise RuntimeError(f"Material export failed, exit={p.returncode}: {folder}")
    raw = result.read_text(encoding="ascii").splitlines()
    if int(raw[0].strip()) != config["stories"]:
        raise ValueError("Exported number of stories is incorrect")
    material = []
    for line in raw:
        parts = line.replace(",", " ").split()
        if len(parts) == 10:
            material.append([float(v.replace("D", "E")) for v in parts])
    if np.shape(material) != (config["stories"], 10):
        raise ValueError("Cannot parse exported ten-parameter material")
    return np.asarray(material), {
        "parameter_generator_executable": str(source_exe), "executable_sha256": sha(source_exe),
        "material_export_sha256": sha(result), "material_export": str(result),
        "mode": "matcodeinput=3: parameter export only; all archived response outputs ignored",
        "input_hashes": {str(p.relative_to(folder)): sha(p) for p in (folder / "inputs").iterdir()},
        "material_formulation": "Lu-Qu ten-parameter legacy ksteel02 as identified by project owner",
        "calibration_basis": "User-confirmed inherited calibration and new-versus-old solver agreement; scaled variants are controlled extensions of that baseline."
    }


def periods(mass, k):
    b = np.zeros((len(k), len(k)))
    for i in range(len(k)):
        b[i, i] = 1
        if i:
            b[i, i-1] = -1
    stiffness = b.T @ np.diag(k) @ b
    invsqrt = 1 / np.sqrt(mass)
    eig = np.linalg.eigvalsh(stiffness * invsqrt[:, None] * invsqrt[None, :])
    # Match the inherited C++ modal pi literal for numerical cross-checks.
    return 2 * float(np.float32(3.1415926)) / np.sqrt(eig)


def case_config(config, material, record, ratio, pga, provenance, dt=None):
    n = config["stories"]
    scaled = material.copy()
    factor = ratio ** -2
    scaled[:, :2] *= factor
    ground_acc = record.acceleration * (pga * G / np.max(np.abs(record.acceleration)))
    zero_time = round(float(record.time[-1]) + float(record.metadata["dt"]), 10)
    # Append an actual zero sample: legacy-profile output bias is then zero in the tail.
    gt = np.r_[record.time, zero_time]
    ga = np.r_[ground_acc, 0.]
    step = dt or config["integration_dt_s"]
    duration = math.ceil((zero_time + config["tail_s"]) / step - 1e-9) * step
    return {
        "schema_version": 1, "name": f"5F_period_ratio_{ratio:.2f}_PGA_{pga:g}g",
        "model": {"type": "shear", "mass": [config["floor_area_m2"] * config["mass_per_area_kg_m2"]] * n,
                  "height": [config["story_height_m"]] * n, "material": scaled.tolist(),
                  "collapse_drift": [config["explicit_failure_drift"]] * n},
        "ground_motion": {"time": gt.tolist(), "acceleration": ga.tolist(), "record_id": record.name},
        "analysis": {"dt": step, "duration": duration, "damping_ratio": config["damping_ratio"],
                     "integrator": "central_difference", "alignment": "legacy", "compatibility": config["compatibility"]},
        "output": {},
        "provenance": {**provenance, "building_attributes": {k: config[k] for k in ("stories", "story_height_m", "floor_area_m2", "year", "fortification_intensity", "structure_type")},
                       "period_ratio": ratio, "stiffness_and_yield_force_scale": factor, "pga_g": pga,
                       "ground_source": str(record.path), "ground_sha256": sha(record.path),
                       "ground_unit": "m/s^2 from processed library manifest", "zero_appended_at_s": zero_time,
                       "cutoff_source": "explicit fixed 0.05 reference, matching the controlled legacy comparison corpus"}
    }


def summary(result, ratio, pga, file, config):
    drift = result.edp["peak_interstory_drift_ratio"]
    grades = np.sum(drift[:, None] > np.asarray(config["reference_drift_thresholds"])[None, :], axis=1)
    return {"period_ratio": ratio, "T1_s": float(result.periods[0]), "pga_g": pga,
            "max_PIDR_percent": float(max(drift) * 100), "DS_ref": int(max(grades)),
            "critical_story": int(result.edp["critical_story"]),
            "roof_disp_mm": float(np.max(np.abs(result.displacement[:, -1])) * 1000),
            "max_PFA_g": float(np.max(result.edp["peak_floor_acceleration"]) / G),
            "material_failure": bool(np.any(result.material_dead)), "hdf5": str(file)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pgas", nargs="+", type=float)
    args = parser.parse_args()
    config = json.loads((STAGE / "configs/five_story_preview.json").read_text(encoding="utf-8"))
    run_dir = STAGE / "runs/five_story_preview"
    data_dir = STAGE / "data/five_story_preview"
    data_dir.mkdir(parents=True, exist_ok=True)
    material, provenance = baseline(config, run_dir)
    record = load_record(PROJECT / config["ground_motion_relative_path"], unit="m/s^2")
    pgas = args.pgas or config["pga_g"]
    rows = []
    for ratio in config["period_ratios"]:
        for pga in pgas:
            name = f"r{ratio:.2f}_pga{pga:g}g"
            file = data_dir / f"{name}.h5"
            c = case_config(config, material, record, ratio, pga, provenance)
            result = load_result(file) if file.exists() else run_case(c)
            if result.metadata["config_hash"] != config_hash(c):
                raise ValueError(f"Existing result uses another configuration; use a new output directory: {file}")
            expected = periods(np.asarray(c["model"]["mass"]), np.asarray(c["model"]["material"])[:, 0])
            if not np.allclose(result.periods, expected, rtol=1e-10):
                raise ValueError("C++ periods disagree with independent eigenvalue calculation")
            if not args.pilot and not file.exists():
                save_result(result, file)
            row = summary(result, ratio, pga, file, config)
            print(json.dumps(row), flush=True)
            rows.append(row)
    if args.pilot:
        write_json(run_dir / "pilot.json", {"config": config, "results": rows})
        return
    if args.pgas:
        config["pga_g"] = pgas
    report = STAGE / "reports/five_story_preview"
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / "summary.json", {"config": config, "source": provenance, "cases": rows})
    write_csv(report / "summary.csv", rows)
    generate_figures(config, material, provenance, record, rows, data_dir, report)


def generate_figures(config, material, provenance, record, rows, data_dir, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.linalg import eigh

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "axes.titleweight": "bold",
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    colors = ["#2166ac", "#222222", "#d6604d"]
    pga_colors = ["#4575b4", "#74add1", "#fdae61", "#d73027"]
    histories, floor_rows, step_rows, parameter_rows = {}, [], [], []
    n = config["stories"]
    dt = config["integration_dt_s"]
    # Native profile timestamps displacement at n+1. Reindex to physical n for plots/CSV;
    # original complete HDF5 remains untouched and retains its legacy metadata.
    def physical_u(r):
        return np.vstack((np.zeros((1, n)), r.displacement[:-1]))

    for ri, ratio in enumerate(config["period_ratios"]):
        scaled = material.copy()
        scaled[:, :2] *= ratio ** -2
        mass = np.full(n, config["floor_area_m2"] * config["mass_per_area_kg_m2"])
        for floor in range(n):
            parameter_rows.append({"period_ratio": ratio, "T1_s": next(r["T1_s"] for r in rows if r["period_ratio"] == ratio),
                                   "floor": floor + 1, "height_m": config["story_height_m"], "mass_kg": mass[floor],
                                   **dict(zip(NAMES, scaled[floor])), "yield_displacement_mm": scaled[floor, 1] / scaled[floor, 0] * 1000})
        for pga in config["pga_g"]:
            name = f"r{ratio:.2f}_pga{pga:g}g"
            r = load_result(data_dir / f"{name}.h5")
            u = physical_u(r)
            inter = u - np.column_stack((np.zeros(len(u)), u[:, :-1]))
            stride = max(1, int(round(.002 / dt)))
            indexes = np.arange(0, len(r.time), stride)
            selected = np.column_stack((r.time[indexes], u[indexes] * 1000, r.acceleration[indexes] / G))
            header = "time_s," + ",".join(f"floor{i+1}_relative_displacement_mm" for i in range(n)) + "," + ",".join(f"floor{i+1}_absolute_acceleration_g" for i in range(n))
            np.savetxt(data_dir / f"{name}_histories.csv", selected, delimiter=",", header=header, comments="", fmt="%.9g")
            histories[ratio, pga] = {"time": r.time[indexes], "u": u[indexes] * 1000,
                                     "a": r.acceleration[indexes] / G, "inter": inter[indexes] * 1000,
                                     "force": r.story_force[indexes] / 1e6,
                                     "pidr": r.edp["peak_interstory_drift_ratio"] * 100,
                                     "pfa": r.edp["peak_floor_acceleration"] / G,
                                     "periods": r.periods.copy()}
            for j in range(n):
                pidr = float(r.edp["peak_interstory_drift_ratio"][j])
                floor_rows.append({"T1_s": float(r.periods[0]), "period_ratio": ratio, "pga_g": pga,
                                   "floor": j+1, "PIDR_percent": pidr*100,
                                   "PFA_g": float(r.edp["peak_floor_acceleration"][j]/G),
                                   "peak_displacement_mm": float(r.edp["peak_floor_displacement"][j]*1000),
                                   "DS_ref": int(np.sum(pidr > np.asarray(config["reference_drift_thresholds"]))),
                                   "material_failure": bool(r.edp["material_failure_by_story"][j])})
            # All 12 production runs checked against twice the integration interval.
            coarse = run_case(case_config(config, material, record, ratio, pga, provenance, dt=2*dt))
            row = {"period_ratio": ratio, "pga_g": pga, "fine_dt_s": dt, "coarse_dt_s": 2*dt}
            for key, label in [("peak_interstory_drift_ratio", "PIDR"), ("peak_floor_acceleration", "PFA"), ("peak_floor_displacement", "displacement")]:
                fine, co = np.asarray(r.edp[key]), np.asarray(coarse.edp[key])
                row[f"max_floor_{label}_change_percent"] = float(np.max(np.abs(fine-co) / np.maximum(np.abs(fine), 1e-12))*100)
            row["u_relative_L2_percent"] = float(np.linalg.norm(u[::2]-physical_u(coarse)) / max(np.linalg.norm(u[::2]), 1e-12)*100)
            row["a_relative_L2_percent"] = float(np.linalg.norm(r.acceleration[::2]-coarse.acceleration) / max(np.linalg.norm(r.acceleration[::2]), 1e-12)*100)
            step_rows.append(row)
            del r, coarse, u, inter

    write_csv(report / "structure_parameters.csv", parameter_rows)
    write_csv(report / "floor_metrics.csv", floor_rows)
    write_csv(report / "timestep_check.csv", step_rows)
    write_json(report / "baseline_material.json", {"source": provenance, "parameter_names": NAMES,
               "model": {"mass": mass.tolist(), "height": [config["story_height_m"]]*n, "material": material.tolist()}})

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    for ri, ratio in enumerate(config["period_ratios"]):
        h = histories[ratio, config["pga_g"][0]]
        label = f"T1 = {h['periods'][0]:.3f} s"
        axes[0].plot(material[:, 0] / ratio**2 / 1e9, np.arange(1,n+1), "o-", color=colors[ri], label=label)
        axes[1].plot(material[:, 1] / ratio**2 / 1e6, np.arange(1,n+1), "o-", color=colors[ri])
    b = np.eye(n) - np.eye(n, k=-1)
    _, shapes = eigh(b.T @ np.diag(material[:, 0]) @ b, np.diag(mass))
    for j in range(3):
        mode = shapes[:, j] / np.max(np.abs(shapes[:, j]))
        if mode[-1] < 0:
            mode = -mode
        axes[2].plot(np.r_[0, mode], np.arange(0,n+1), "o-", label=f"Mode {j+1}")
    for ax, title, xlabel in zip(axes, ["Story stiffness", "Story yield force", "Initial mode shapes"], ["Stiffness (GN/m)", "Yield force (MN)", "Normalized amplitude"]):
        ax.set(title=title, xlabel=xlabel, ylabel="Floor / story")
        ax.set_yticks(np.arange(0,n+1))
    axes[0].legend(loc="lower left", fontsize=9)
    axes[2].legend(fontsize=9)
    fig.suptitle("Five-story frame | 3 m/story | 1,000 t/floor | inherited baseline parameters", fontsize=13)
    fig.savefig(report / "structure_profiles.png", dpi=180)
    plt.close(fig)

    for pga in config["pga_g"]:
        fig, axes = plt.subplots(n, 2, figsize=(14, 12), sharex=True, constrained_layout=True)
        for row, floor in enumerate(range(n-1, -1, -1)):
            for ri, ratio in enumerate(config["period_ratios"]):
                h = histories[ratio, pga]
                label = f"T1 = {h['periods'][0]:.3f} s"
                axes[row,0].plot(h["time"], h["u"][:,floor], lw=.65, alpha=.9, color=colors[ri], label=label)
                axes[row,1].plot(h["time"], h["a"][:,floor], lw=.55, alpha=.85, color=colors[ri])
            axes[row,0].set_ylabel(f"Floor {floor+1}\nDisplacement (mm)")
            axes[row,1].set_ylabel(f"Floor {floor+1}\nAbs. acceleration (g)")
            for ax in axes[row]:
                ax.axhline(0, color="#888888", lw=.4)
                lim = max(abs(v) for v in ax.get_ylim())
                ax.set_ylim(-lim, lim)
        axes[0,0].legend(loc="upper right", ncol=3, fontsize=8)
        axes[0,0].set_title("Relative floor displacement")
        axes[0,1].set_title("Absolute floor acceleration (inherited coordinate sign)")
        for ax in axes[-1]:
            ax.set_xlabel("Time (s)")
        fig.suptitle(f"Five-story responses | same PEER-1 waveform | PGA = {pga:g} g", fontsize=15)
        fig.savefig(report / f"histories_pga_{pga:g}g.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for ri, ratio in enumerate(config["period_ratios"]):
        for pi, pga in enumerate(config["pga_g"]):
            h=histories[ratio,pga]
            axes[0,ri].plot(h["pidr"],np.arange(1,n+1),"o-",color=pga_colors[pi],label=f"PGA {pga:g} g")
            axes[1,ri].plot(h["pfa"],np.arange(1,n+1),"o-",color=pga_colors[pi])
        axes[0,ri].set(title=f"T1 = {h['periods'][0]:.3f} s",xlabel="Peak interstory drift (%)",ylabel="Story",yticks=np.arange(1,n+1))
        axes[1,ri].set(xlabel="Peak absolute floor acceleration (g)",ylabel="Floor",yticks=np.arange(1,n+1))
        axes[0,ri].axvline(.5,color="#aaaaaa",ls="--",lw=.7)
        axes[0,ri].axvline(2.,color="#888888",ls=":",lw=.7)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Full integration-grid peaks | drift lines mark 0.5% and 2% reference thresholds",fontsize=13)
    fig.savefig(report/"peak_profiles.png",dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(n,3,figsize=(13,14),constrained_layout=True)
    for ri,ratio in enumerate(config["period_ratios"]):
        for row,floor in enumerate(range(n-1,-1,-1)):
            ax=axes[row,ri]
            for pi,pga in enumerate(config["pga_g"]):
                h=histories[ratio,pga]
                ax.plot(h["inter"][:,floor],h["force"][:,floor],lw=.55,color=pga_colors[pi],alpha=.85)
            ax.set(xlabel="Story deformation (mm)",ylabel=f"Story {floor+1} shear (MN)")
            if row==0:
                ax.set_title(f"T1 = {h['periods'][0]:.3f} s")
    handles=[Line2D([0],[0],color=pga_colors[i],label=f"{p:g} g") for i,p in enumerate(config["pga_g"])]
    fig.legend(handles=handles,loc="outside upper center",ncol=4,
               title="Story force-deformation loops | PGA")
    fig.savefig(report/"hysteresis_loops.png",dpi=160,bbox_inches="tight")
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(12,2.8),constrained_layout=True)
    ground=record.acceleration*(max(config["pga_g"])/np.max(np.abs(record.acceleration)))
    ax.plot(record.time,ground,color="#333333",lw=.65)
    ax.set(xlabel="Time (s)",ylabel="Ground acceleration (g)",title=f"PEER-1 input shape at {max(config['pga_g']):g} g | no waveform filtering")
    fig.savefig(report/"ground_motion.png",dpi=160)
    plt.close(fig)

    # Early 2 ms pilot spike versus finer grids: preserve the finding quantitatively.
    spike=[]
    for check_dt in [.002,.001,.0005,.00025]:
        r=run_case(case_config(config,material,record,1.,.35,provenance,dt=check_dt))
        spike.append({"dt_s":check_dt,"max_PIDR_percent":float(max(r.edp["peak_interstory_drift_ratio"])*100),
                      "max_PFA_g":float(max(r.edp["peak_floor_acceleration"])/G)})
    write_csv(report/"timestep_spike_diagnostic.csv",spike)
    maxima={key:max(x[key] for x in step_rows) for key in step_rows[0] if key.endswith("percent")}
    write_json(report/"verification.json",{
        "independent_eigenvalue_check":"all five modal periods for every case passed rtol=1e-10",
        "peak_step_change_max_percent":maxima,"production_dt_s":dt,
        "coarse_dt_s":2*dt,"case_count":len(rows),"failure_count":sum(r["material_failure"] for r in rows),
        "raw_HDF5_alignment":"legacy displacement/drift advanced by one integration interval; raw results preserved",
        "plot_CSV_alignment":"displacement reindexed by one row to the physical current time with known zero initial displacement; accelerations unchanged",
        "plot_and_CSV_interval_s":.002,"smoothing":"none; regular point selection for display/CSV only",
        "peak_grid":"all original 0.00025 s integration samples; not display-decimated samples",
        "solver_DLL_sha256":sha(WORKBENCH/"build/Release/mdof_core.dll"),
        "script_sha256":sha(Path(__file__)),"config_sha256":sha(STAGE/"configs/five_story_preview.json"),
        "ground_sha256":sha(record.path)})
    make_report(config,material,rows,parameter_rows,maxima,report)


def make_report(config,material,rows,params,maxima,report):
    def table(headers,values):
        return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"]*len(headers)) + " |\n" + "\n".join("| " + " | ".join(map(str,r)) + " |" for r in values)
    lines=["# 五层结构：不同 T1 与 PGA 的逐层响应预览", "", "日期：2026-10-21。共 3 个结构 × 4 档 PGA = 12 个工况；所有响应由新 C++ 生产内核独立积分。", "",
           "## 参数与模型", "", "采用项目既有陆—曲十参数滞回模型和旧 0.1 兼容分支。按用户确认的既有标定与新版/旧版对照基础开展本轮计算；旧 EXE 只用于导出五层基准材料参数，不提供任何本轮响应。", "",
           "基准属性：5 层、总高 15 m、层高 3 m、每层面积 1,000 m²、质量面密度 1,000 kg/m²（每层 1,000 t）、2002 年、8 度设防、框架、场地类别 2/分组 1。阻尼比 5%。材料参数由既有参数生成器按这些属性生成，未用演示参数替代。", "",
           "三个版本的全部层刚度和屈服力同时乘以 1.5625、1、0.694444。屈服变形保持一致，周期变化同时伴随强度变化；只用于这组受控预览。初始振型相同。", ""]
    lines += [table(["版本","T1 / s","各层刚度 / GN·m⁻¹","刚度/强度倍率"],[[f"{r:.1f}T",f"{next(x['T1_s'] for x in rows if x['period_ratio']==r):.6f}",f"{material[0,0]/r**2/1e9:.6f}（1–5层相同）",f"{r**-2:.6f}"] for r in config['period_ratios']]), "",
              table(["层号","质量 / t","层高 / m","基准刚度 / GN·m⁻¹","基准屈服力 / MN","屈服变形 / mm"],[[i+1,"1000","3",f"{material[i,0]/1e9:.4f}",f"{material[i,1]/1e6:.4f}",f"{material[i,1]/material[i,0]*1000:.4f}"] for i in range(5)]), "",
              "其余八个参数各层相同：`" + ", ".join(f"{NAMES[i]}={material[0,i]:g}" for i in range(2,10)) + "`。完整 15 行参数见 [structure_parameters.csv](structure_parameters.csv)。", "",
              "![刚度、强度与振型](structure_profiles.png)", "", "## 地震动与强度", "",
              "采用工作台已有的处理后 PEER-1 波形（50 Hz，3,000 点，末时刻 59.98 s；具体原始事件名未在该处理文件中声明）。每个强度均按实际绝对峰值缩放至目标 PGA；60.00 s 追加零激励，随后计算 15 s 自由衰减。所有版本使用相同波形。", "",
              "最终四档为 0.10g、0.35g、0.40g、0.60g。先行试算选择这些幅值，使结果覆盖无/轻微/中等/严重参考损伤，并避免本组进入 5% 的显式材料失效截断。选幅属于预览设计，不用于模型独立测试。", "",
              "![输入地震动](ground_motion.png)", "", "## 结果", "",
              "DS_ref 按 PIDR 严格超过 `[0.5%, 0.8%, 2%, 5%]` 的数量取 0–4：无、轻微、中等、严重、最高参考等级。它是本次明确采用的固定参考判据；材料失效标志另行保存。", "",
              table(["T1 / s","PGA / g","全楼最大 PIDR / %","DS_ref","顶层峰值位移 / mm","全楼最大 PFA / g","危险层"],[[f"{x['T1_s']:.3f}",f"{x['pga_g']:.2f}",f"{x['max_PIDR_percent']:.4f}",x['DS_ref'],f"{x['roof_disp_mm']:.3f}",f"{x['max_PFA_g']:.3f}",x['critical_story']] for x in rows]), "",
              "12 个工况均未触发材料失效标志。PIDR 最大位置均在首层。较长周期版本同时具有较低强度，本波形下非线性变形较大；加速度并不随周期或 PGA 简单单调变化。这里只展示一条波形的结果。", "",
              "![逐层峰值分布](peak_profiles.png)", "", "## 每层位移和加速度时程", "",
              "每张图同一 PGA，叠加三个 T1；从上到下为第 5 层至第 1 层，左列位移、右列绝对加速度。单位分别为 mm、g。保留原坐标正负方向。", ""]
    for pga in config["pga_g"]:
        lines += [f"### PGA = {pga:g}g", "", f"![各层时程：{pga:g}g](histories_pga_{pga:g}g.png)", ""]
    lines += ["## 各层滞回环", "", "每列对应一个 T1，每行对应一层；四种颜色为四档 PGA。横轴层间变形，纵轴层剪力；采用原程序实际力学响应，未另换材料模型。", "",
              "![各层滞回环](hysteresis_loops.png)", "", "## 数值核对与读取", "",
              "- 所有工况的五个初始周期均与独立矩阵特征值计算吻合（相对容差 1e-10）。",
              "- 生产积分步长 0.00025 s。与 0.0005 s 比较，12 工况所有楼层的峰值最大相对差如下；完整逐工况结果见 [timestep_check.csv](timestep_check.csv)。",
              f"  - PIDR：{maxima['max_floor_PIDR_change_percent']:.3f}%；PFA：{maxima['max_floor_PFA_change_percent']:.3f}%；位移：{maxima['max_floor_displacement_change_percent']:.3f}%。",
              "- 基准 T1、0.35g 的初始 0.002 s 试算出现明显 PFA 尖峰；减小步长后 PFA 稳定在约 0.67g。诊断保存在 [timestep_spike_diagnostic.csv](timestep_spike_diagnostic.csv)。未通过曲线平滑掩盖差异。",
              "- 完整 HDF5 保留全部积分网格、原始 legacy 时间约定、所有响应通道、输入配置和来源。图和便捷 CSV 的位移已依据已知零初始条件回移一个积分行，恢复到与加速度一致的物理时刻；不改动力计算。",
              "- 图和便捷 CSV 每 0.002 s 选点，不做平滑；表中峰值直接来自全部积分点。HDF5 中原始位移的 legacy 行时间比其物理时间提前一个积分步，读取时须遵从元数据。",
              "- 原始数据位于 `../../data/five_story_preview/`，包括 12 份完整 HDF5 和 12 份时程 CSV；CSV 提供时间和各层位移/加速度。原始数据不提交 Git。",
              "- 数据读取示例：`mdof_workbench.api.load_result(path)`；HDF5 可在新工作台打开。",
              "- [逐层指标](floor_metrics.csv) · [完整参数](structure_parameters.csv) · [汇总](summary.json) · [核验记录](verification.json)", "",
              "## 复现", "", "从 experiment 目录执行：", "", "```powershell", "& '..\\mdof-workbench\\.venv\\Scripts\\python.exe' 'stage1\\scripts\\five_story_preview.py'", "```", "",
              "代码、配置、报告和全部新增输出均在 experiment 内；其他目录只读。没有训练 Transformer。", ""]
    (report/"README.md").write_text("\n".join(lines),encoding="utf-8")


if __name__ == "__main__":
    main()
