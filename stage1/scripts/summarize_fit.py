"""Summarize the remote fit campaign, including data checks and derived EDPs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np

STAGE=Path(__file__).resolve().parents[1]
RUNS=["poc-fit24-patch60-seed20261021","poc-fit24-patch60-refine-seed20261021",
      "poc-fit24-patch60-balanced-seed20261021","poc-fit24-patch60-balanced-long-seed20261021"]


def main():
    reports=[json.loads((STAGE/"reports"/r/"result.json").read_text()) for r in RUNS]
    manifest_path=STAGE/"data/poc_fit24/manifest.json"
    manifest=json.loads(manifest_path.read_text())
    cases=[x for x in manifest["cases"] if x["status"]=="complete"]
    sampling_errors=[];period_errors=[];pidrs=[];structures=[]
    for case in cases:
        path=manifest_path.parent/case["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==case["npz_sha256"]
        with np.load(path,allow_pickle=False) as x:
            assert all(np.isfinite(x[k]).all() for k in x.files)
            n=case["stories"]
            assert x["displacement"].shape==x["acceleration"].shape==(3751,n)
            assert x["ground"].shape==(3751,1)
            theta=np.diff(x["displacement"],axis=1,prepend=np.zeros((3751,1)))/x["story_heights"]
            peak=float(np.abs(theta).max());pidrs.append(peak)
            sampling_errors.append(abs(peak-case["max_PIDR_integral"])/case["max_PIDR_integral"])
        source=case["source_config"]["model"]
        material=np.asarray(source["material"])
        mass=np.asarray(source["mass"])
        b=np.eye(n)-np.eye(n,k=-1)
        eig=np.linalg.eigvalsh((b.T@np.diag(material[:,0])@b)/np.sqrt(mass[:,None]*mass[None,:]))
        period=2*float(np.float32(3.1415926))/np.sqrt(eig[0])
        period_errors.append(abs(period/case["T1_s"]-1))
        if case["pga_g"]==.1:
            structures.append({"stories":n,"T1_s":case["T1_s"],"period_ratio":case["period_ratio"],
                               "mass_kg":source["mass"],"height_m":source["height"],
                               "stiffness_N_per_m":material[:,0].tolist(),"yield_force_N":material[:,1].tolist()})
    latest=reports[-1]
    thresholds=np.asarray([.005,.008,.02,.05])
    edps=[]
    with np.load(STAGE/"runs"/RUNS[-1]/"predictions.npz",allow_pickle=False) as f:
        for i,name in enumerate(f["case_ids"]):
            n=int(f["floor_mask"][i].sum())
            row={"case_id":str(name)}
            for prefix,uk,ak in [("pred","displacement","acceleration"),("true","truth_displacement","truth_acceleration")]:
                u=f[uk][i,:,:n];a=f[ak][i,:,:n]
                theta=np.diff(u,axis=1,prepend=np.zeros((len(u),1)))/3.
                pidr=np.abs(theta).max(0);pfa=np.abs(a).max(0)
                row[prefix]={"PIDR":pidr.tolist(),"PFA_m_s2":pfa.tolist(),
                             "critical_story":int(pidr.argmax()+1),"DS_ref":int((pidr.max()>thresholds).sum())}
            edps.append(row)
    quality={"complete":len(cases),"excluded":len(manifest["cases"])-len(cases),
             "finite_and_shape_checks":"PASS","max_relative_T1_eigen_check_error":max(period_errors),
             "PIDR_50Hz_relative_to_integral_peak_error_max":max(sampling_errors),
             "PIDR_50Hz_relative_to_integral_peak_error_median":float(np.median(sampling_errors)),
             "PIDR_50Hz_range":[min(pidrs),max(pidrs)],"structures":structures}
    derived={"scope":"training-set only; reference drift grades are not FEMA P-58 fragilities",
             "thresholds":thresholds.tolist(),
             "building_DS_ref_accuracy":float(np.mean([x["true"]["DS_ref"]==x["pred"]["DS_ref"] for x in edps])),
             "critical_story_accuracy":float(np.mean([x["true"]["critical_story"]==x["pred"]["critical_story"] for x in edps])),
             "cases":edps}
    for filename,value in [("poc_fit24_quality.json",quality),("poc_fit24_derived.json",derived)]:
        (STAGE/"reports"/filename).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    s=latest["summary"]
    lines=["# Stage 1 POC 当前结果","",f"小样本拟合：**{latest['status']}**；整个 Stage 1：**PARTIAL**（独立地震动测试与三种子复核尚未完成）。","",
           "全部仿真、数据处理、模型检查、训练和指标计算在 RTX 5090 服务器执行。本地只编辑代码、同步和查看报告。","",
           "## 数据与模型","","24 个有效算例：3/5/8/12 层 × 3 档周期 × 0.1/0.35 g。全部共用同一条基础地震动，仅用于训练集拟合。",
           "T1 由新 C++ 求解器计算，质量报告另用特征值核对。改变周期时刚度和屈服力同步按比例缩放，保留完整逐层参数；这不是单独改变周期的因果试验。",
           "","FiLM Transformer + 共享楼层查询解码器联合训练；patch=60、128 维、4 层、1,116,792 参数。输出每层位移和绝对加速度；PIDR/PFA 与参考等级从响应派生。","",
           "## 拟合结果","","| 实验 | 位移相对 L2 中位数 | 加速度相对 L2 中位数 | 拟合状态 |","| --- | ---: | ---: | --- |"]
    labels=["初始训练","降低学习率","逐算例归一化","延长均衡拟合"]
    for label,r,run in zip(labels,reports,RUNS):
        lines.append(f"| [{label}]({run}/README.md) | {r['summary']['u_median_relative_l2']:.2%} | {r['summary']['a_median_relative_l2']:.2%} | {r['status']} |")
    lines += ["",f"最终位移 P90 / 最差：{s['u_p90_relative_l2']:.2%} / {s['u_worst_relative_l2']:.2%}；加速度 P90 / 最差：{s['a_p90_relative_l2']:.2%} / {s['a_worst_relative_l2']:.2%}。",
              f"逐算例先对楼层取中位数、再对算例取中位数的 PIDR/PFA 相对误差：{s['PIDR_median_relative_error']:.2%} / {s['PFA_median_relative_error']:.2%}。",
              f"参考全楼损伤等级一致率：{derived['building_DS_ref_accuracy']:.2%}；最大漂移楼层一致率：{derived['critical_story_accuracy']:.2%}。这些均为训练集指标。",
              "",f"最终独立精调轮次：{latest['best_epoch']}；本轮训练时间 {latest['wall_seconds']:.1f} s；峰值 PyTorch 分配显存 {latest['peak_gpu_memory_gb']:.3f} GiB。继承的训练轮次和权重哈希见 result.json，不能只用本轮时长代表全部训练开销。",
              "",f"![最终学习曲线]({RUNS[-1]}/learning_curve.png)","",f"![五层逐层时程]({RUNS[-1]}/fit_05F.png)","",
              "## 核验与边界","",f"24 个文件哈希、有限值和维度检查通过；独立 T1 核验最大相对误差 {max(period_errors):.3g}。",
              f"50 Hz 保存时程的最大 PIDR 与原积分网格峰值相比，逐算例相对差异的中位数为 {np.median(sampling_errors):.2%}、最大为 {max(sampling_errors):.2%}。模型 EDP 指标均在相同 50 Hz 网格计算。",
              "","7 项模型检查通过。逐算例归一化降低了小响应被大响应主导的问题；只修改训练损失，推理不需要响应真值。",
              "","本轮尚未检验未见地震动、未见结构、不同结构类型、变采样率或 FEMA P-58 损失预测。参考损伤等级仅使用显式漂移阈值，不代替构件易损性模型。",
              "","## 下一步","","保持当前模型作为基线；按 OKR 生成多地震动小数据集，按基础地震动分组划分训练/验证/测试，先做一个固定种子。独立测试通过并完成三种子复核后，再判定整个 Stage 1 PASS 并扩大数据。","",
              "数据核验：[poc_fit24_quality.json](poc_fit24_quality.json)。逐层峰值与参考等级：[poc_fit24_derived.json](poc_fit24_derived.json)。"]
    (STAGE/"reports/poc_result.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"fit_status":latest["status"],"stage1_status":"PARTIAL","summary":s,
                      "quality":{k:v for k,v in quality.items() if k!="structures"},
                      "reference_grade_accuracy":derived["building_DS_ref_accuracy"]}),flush=True)


if __name__=="__main__":main()
