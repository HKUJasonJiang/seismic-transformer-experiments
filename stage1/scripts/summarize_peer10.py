"""Write the current POC conclusion from frozen reports; does not rerun test inference."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path

STAGE=Path(__file__).resolve().parents[1]


def main():
    config=json.loads((STAGE/"configs/peer10_train.json").read_text())
    report=STAGE/"reports"/config["run_id"]
    r=json.loads((report/"result.json").read_text());q=json.loads((report/"data_quality.json").read_text())
    training=json.loads((report/"training_result.json").read_text());curve=json.loads((report/"curve.json").read_text());last=curve[-1]
    test=r["splits"]["test"]["summary"];nonlinear=test["DS_gt_2"];weak=test["DS_le_2"]
    under=sum(x["pred_DS_ref"]<x["DS_ref"] for x in r["splits"]["test"]["cases"] if x["DS_ref"]>2)
    lines=["# PEER-10 结论：查询接口通过，泛化精度未通过" if r["status"]!="PASS" else "# PEER-10 结论","",
           "## 已完成","","新 C++ 生成 337 个有效算例：3/5/8 层、各三档周期、10 条基础波。训练 7 条、验证 1 条、测试 2 条；同一基础波的缩放版本没有跨组。90 个结构—地震动组合均覆盖 DS>2。",
           f"DS>2 按既有阈值定义为 PIDR>2%。该子集屈服变形比范围 {q['nonlinear_ductility_range'][0]:.1f}–{q['nonlinear_ductility_range'][1]:.1f}；保存了逐层剪力—变形历史。DS≤2 不代表线性。",
           "",f"{q['simulation_trials']} 次生成/搜索试算中有 {q['failed_trials']} 次失败试算，均未用作普通响应标签；保留的 {q['cases_checked']} 个数据文件通过哈希、有限值、维度、输入 PGA 和周期核验。",
           f"3/5/8 层各取一个预先指定的强非线性训练算例，将积分步长减半。最大 PIDR 差异 {max(x['PIDR_relative_difference'] for x in q['refined_step_cases']):.2%}；最大 PFA 差异 {max(x['PFA_relative_difference'] for x in q['refined_step_cases']):.2%}。这只核验了三个工况，不是对全部地震动的收敛证明。",
           "","## 查询与预测是两个判定","",f"69 个未见波测试算例，共 {r['query_check']['single_floor_queries']} 次独立楼层查询通过；乱序查询也通过。单层与整楼的对应输出一致，支持混合 3/5/8 层的调用。",
           f"但 DS>2 的 {nonlinear['count']} 个未见波算例，时程相对 L2 中位数为：位移 **{nonlinear['u_median_relative_l2']:.2%}**、加速度 **{nonlinear['a_median_relative_l2']:.2%}**；PIDR/PFA 楼层误差中位数为 **{nonlinear['PIDR_floor_median_relative_error']:.2%} / {nonlinear['PFA_floor_median_relative_error']:.2%}**。",
           f"参考损伤等级一致率只有 {nonlinear['DS_ref_accuracy']:.2%}，{under}/{nonlinear['count']} 个强非线性测试算例被低估。当前模型不能用于可靠的非线性损伤判断。",
           "","| DS>2 未见波 | 算例 | 位移 L2 中位数 | 加速度 L2 中位数 |","| --- | ---: | ---: | ---: |"]
    for n in (3,5,8):
        s=test["by_stories"][str(n)]["DS_gt_2"]
        lines.append(f"| {n} 层 | {s['count']} | {s['u_median_relative_l2']:.2%} | {s['a_median_relative_l2']:.2%} |")
    lines += ["","## 不能归结为单一原因","",
              f"DS≤2 的测试误差也较大：位移 {weak['u_median_relative_l2']:.2%}、加速度 {weak['a_median_relative_l2']:.2%}。问题并不限于高损伤子集。",
              f"本轮训练在 {training['last_epoch']} epoch 停止，验证选择 {r['best_epoch']} epoch。最终训练轮次的训练误差为位移 {last['train']['all']['u_median_relative_l2']:.2%}、加速度 {last['train']['all']['a_median_relative_l2']:.2%}，仍未达到双通道 10% 的拟合门槛。验证选择的模型与最终训练轮次不同，完整记录保留在 curve.json。",
              "现有证据表明单波拟合成绩不能支撑未见地震动泛化。还不能只把原因归到楼层查询 B；需要分别定位时序编码、楼层解码和优化容量。测试集已使用，后续模型比较应依靠训练/验证，正式确认要重新保留测试记录。",
              "","## 下一步顺序","","先在训练分区对覆盖 3/5/8 层的强非线性子集做拟合诊断；再隔离编码器与查询解码器的误差来源，比较带局部时间信息或 FFT/FNO 分支的编码器。保持当前十波数据作为诊断集，先不扩大生成量，也不把当前结果包装成非线性泛化已解决。",
              "","[完整指标与逐层曲线](README.md) · [数据与步长核验](data_quality.json)","",
              "![训练与验证差距](learning_curve.png)","","![强非线性真值滞回](reference_hysteresis.png)"]
    (report/"CONCLUSION.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":r["status"],"DS_gt_2_test":nonlinear["count"],"underestimated":under,"training_epoch":training["last_epoch"],"selected_epoch":r["best_epoch"]}),flush=True)


if __name__=="__main__":main()
