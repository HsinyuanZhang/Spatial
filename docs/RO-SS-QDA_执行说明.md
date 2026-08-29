# RO-SS-QDA — Grok Build 执行说明

面向仓库工作副本 `/home/xinyuan/SNN_SpikeSorting/Spatial`（远程 `HsinyuanZhang/Spatial`，分支 `shape-orthogonal-descriptor`）。

工作名 STAR-Mem。本文是**执行规格**，不是论文。先静态证伪，不要做在线更新，不要做 CAM/RTL，不要量化。

讨论底稿：`docs/STAR-Mem_优雅实现_讨论稿.md`（若无，以本文冻结字段为准）。Mapping Row v0 已停，不要改那些否定 runner 装活。

---

## 0. 一句话任务

实现并跑通一组冻结对照，回答：

> 在观测锁死为 7 电极正负峰（POSNEG）、坐标系锁死为 relative-offset 时，**正则化全协方差 QDA** 能否比 **同一 \(z\) 上的 mean+L1/L2** 更好地分开同一 home 电极上的单元？改进若存在，是来自跨维相关，还是只来自方差 / 体积项？

过门才谈把模板改成 \((\mu,\Sigma)\)。不过门就停。不要扫分位、mask、原型数、HDC、波形。

---

## 1. 边界（必须遵守）

- 只改 `Spatial/`。不要碰 `SpikeSortingSNN`、`iBCI`、父仓库其它目录。
- **观测：** 每个事件只许用邻域 \(K=7\) 的正向峰值 \(q^+\) 和负向谷值 \(q^-\)（绝对值）。不要先合成 P2P 当主特征（P0 历史复现除外）。不要用整段波形、dyadic、pair-tap、16-sample 前缀。
- **分类：性能优先，float。** 不要 5-bit、不要行宽、不要 SRAM/CAM、不要 0.99/4× 流量门当选分类器的标准。候选召回–\(\lvert C\rvert\) 曲线可作诊断。
- **禁止重开：** 区间 / 5/95 盒、二进制 mask、\(\{1,2,4\}\) 权、\(P\le 4\) 多原型、Range-CAM 早退、shift-min 多 \(S\)、在线 EMA/SUSPECT/开簇/合并、HDC bundle 当主分类器。
- 现有检索实验默认：真值事件、真值行、整段零相位滤波。本刀保持这个 isolation，写成 offline component isolation。
- 开发集（先跑、冻结配置）：与 Mapping Row / 既有 DAC 四条 60 s 记录相同
  - `drift16c_600s_11`
  - `static16c_600s_11`
  - `rec_v1_units10_snr5_seed202601`
  - `rec_v1_units20_snr5_seed202601`
- 时间划分（冻结）：前 50% fit / 中 25% calibration / 后 25% test。相同时间戳必须留在同一段。
- 没有 `--confirm` 不得动确认集。

---

## 2. 描述子（所有新臂共用）

对每个事件：

1. 锚点 \(h_e=\arg\max_i \max(q_i^+, q_i^-)\)。
2. 按电极**真实物理坐标**把 7 个邻居填进固定 relative-offset 槽（不要 KNN 距离顺序）。正、负分别排列，得到

$$
x=\mathrm{RO}(q^+,q^-)\in\mathbb{R}^{14}_{\ge 0}.
$$

一维探针用主轴 offset。二维探针用固定物理 stencil。槽位上「电极不存在」和「存在但安静」要能区分（实现里用 mask 或 NaN→0 并单列记录 absent；分类器主路径把 absent 当 0，但不要把 absent 当成可调超参）。

3. 公共尺度必须是 \(\ell_1\)，正负除同一个 \(m\)：

$$
m=\|x\|_1,\qquad
u=\frac{x}{m+\varepsilon},\qquad
z=\bigl[u,\;\log(m+\varepsilon),\;\mathrm{COM}\bigr].
$$

COM 由同一组 POSNEG 幅度和电极坐标计算。一维探针只留一个 COM 坐标。**不要**用 \(\|x\|_\infty\) 当主尺度（P1b 可选对照）。

4. 标准化：只用 **fit** 集的逐维均值和标准差。calibration / test 不得重估。

复用 `algorithms/spatial_footprint.py` 里已有的 relative-offset / POSNEG / COM。缺 gather-by-slot 就在旁边加，不要重写 loader。

---

## 3. 分类器

### P0（历史复现）

现有 5-bit 或 float 的 COM+P2P **mean + L1**（与 `mapping_row` J0 / 既有 L1 源对齐）。只确认数据和 split 没漂。允许与 J0 召回有文档写明的舍入差。

### P1（新的最强 mean 基线）

同一 \(z\) 上：

- mean + L1
- mean + L2

都是 float。在 **calibration** 上选较强的一个，冻结到 test。这是 P2 的正式对照（P1）。

可选 P1b：同一套，但 \(m=\|x\|_\infty\)。只报告，不当正式基线。

### P2（主提案：RO-SS-QDA）

每个单元 \(k\) 在 fit 上估 \(\mu_k=\mathbb{E}[z\mid k]\)、\(\widehat{\Sigma}_k=\mathrm{Cov}[z\mid k]\)。

$$
\widetilde{\Sigma}_k=(1-\lambda)\widehat{\Sigma}_k+\lambda\Sigma_{\mathrm{pool}}+\eta I
$$

- \(\Sigma_{\mathrm{pool}}\)：全体单元的 pooled within-class covariance（fit）。
- \(\eta I\)：只为数值稳定，固定小常数（如 \(10^{-6}\)，写进 plan）。
- \(\lambda\)：**全局一个值**，只许从 \(\{0.25,0.5,0.75\}\) 在 calibration 上选一次（目标：assignment accuracy + 同 home margin 的 p10）。选完冻结到 test。不为每条记录、每个单元另选。

先验均等，不按 firing rate。

$$
s_k(z)=-\frac12(z-\mu_k)^\top\widetilde{\Sigma}_k^{-1}(z-\mu_k)-\frac12\log\bigl|\widetilde{\Sigma}_k\bigr|
$$

$$
\hat y=\arg\max_k s_k(z)
$$

**必须保留 \(\log|\Sigma|\)。** 只算马氏距离会偏袒宽分布单元。

样本过少的单元（fit 事件数 \(< d+5\) 或协方差秩不足）：该单元强制 \(\lambda=1\)（只用 pooled），并在结果里点名。不要静默奇异。

### P3（机制对照）

完全相同的 \(z\)、split、均值、先验、\(\lambda\)、\(\eta\)，但把 \(\widehat{\Sigma}_k\) 和 \(\Sigma_{\mathrm{pool}}\) 的非对角清零（对角 QDA）。

P2 若不优于 P3 的同 home 指标，不准写「跨通道相关」。

### P2b（归因）

同一 \(z\)，\(\lambda=1\)（纯共享协方差 QDA，所有单元共用 \(\Sigma_{\mathrm{pool}}\)）。用来区分：改进来自 per-unit \(\Sigma\) 还是 shared \(W\)。

不要在本刀实现局部 LDA 投影、NCA、MLP。

### D1（并行诊断，不挡 P2）

对每个同 home 单元对，用 fit 均值报：

- 幅度形状 \(u\) 的余弦相似度
- 极性 profile 距离（例如 \(\rho=(q^+-q^-)/(q^++q^-+\varepsilon)\) 的 L2）
- \(\|\Delta_i-\Delta_j\|\)，\(\Delta=\mathrm{COM}(q^+)-\mathrm{COM}(q^-)\)
- 该对上有监督 LDA / 线性可分 oracle 精度（只作上界）

三种读法见讨论稿 0.2。D1 解释 P2，不批准 P2。

---

## 4. 必须报告的指标

对每条记录 + HJ/MEA 家族合计（写明 event-weighted）：

- assignment accuracy（MAP / argmin）
- 每单元 recall、最弱单元 recall
- 间隔（QDA 用分数差；mean 臂用距离差，**不要跨度量比绝对值**）

$$
m_e=s_{\mathrm{true}}-\max_{j\ne y}s_j,\qquad
m_e^{\mathrm{home}}=s_{\mathrm{true}}-\max_{j\ne y,\,h_j=h_y}s_j
$$

  报 \(m_e\) 与 \(m_e^{\mathrm{home}}\) 的 p10、中位、\(P(m>0)\)。
- 错误落在同 home / 邻 home / 远处的比例
- 同 home pairwise 混淆（可按记录）
- P2 选出的 \(\lambda\)
- 各单元 fit 计数；哪些被强制 \(\lambda=1\)

可选诊断：把分数阈值化成候选集 \(C\) 的召回–\(\lvert C\rvert\) 曲线。**不是过门条件。**

---

## 5. P2 过门（开发四记录，性能）

1. HJ **和** MEA 的 assignment accuracy 比冻结 P1 高 \(\ge 2\) pp，**或** 两边同 home 错误都相对降 \(\ge 20\%\)。
2. 两个家族的 \(m_e^{\mathrm{home}}\) 的 **p10** 都提高，不能只涨均值。
3. 没有一条记录比 P1 掉超过 1 pp；最弱单元 recall 掉不超过 2 pp。
4. P2 相对 P3，两个家族的同 home 指标都一致更好。否则只能写「高斯 / 方差」，不能写「跨维相关」。
5. 同 home 子集必须单独变好。不能全靠 COM 排除远处单元。

读结果（写进 `docs/ro_ss_qda_results.md`）：

| 格局 | 结论 |
|---|---|
| P2 > P3 ≈ P1 | 相关结构成立；模板对象可改为 \((\mu,\widetilde{\Sigma},\log\|\Sigma\|)\)。**仍不实现在线。** |
| P2 ≈ P3 > P1 | 高斯 / 体积惩罚有效，相关故事不成立 |
| P2 ≈ P1，P2b 也不好，D1 oracle 高 | 分类器弱；可另开受限共享 LDA 当 POSNEG 上界，本刀不要做 |
| 都贴 chance | POSNEG 对部分同电极单元不是充分观测；下一分支才允许最小时间量 |

MAP 涨了但 0.99 候选集仍肥：只许写「POSNEG 能更好分类」，不许写 4× 流量。

---

## 6. 实现落点

新文件，不要把否定实验改写成还活着：

- `algorithms/ro_ss_qda.py`：`make_descriptor`、标准化、pooled shrinkage、QDA 分数、同 home margin
- `experiments/run_ro_ss_qda.py`：P0/P1/P2/P3/P2b + D1，一条命令跑完四条开发记录
- `tests/test_ro_ss_qda.py`
- `docs/ro_ss_qda_plan.md`：抄本文冻结字段
- `docs/ro_ss_qda_results.md`：总表 + 过门 + 按第 5 节读结果
- `docs/ro_ss_qda_provenance.md` + sha256

必须复用：

- 现有 data loader、split、GT 事件对齐、collision-safe 时间戳
- `spatial_footprint` 的 relative-offset / COM / POSNEG
- 现有评价记账能复用的部分；新的 \(m_e^{\mathrm{home}}\) 另加，不要改旧 CSV 列语义

CLI 示例：

```bash
python -m Spatial.experiments.run_ro_ss_qda \
  --pilot --duration 60 --quiet \
  --arms P0,P1,P2,P3,P2b,D1
```

默认 `--pilot` = 四条开发记录。

单元测试至少：

1. \(q^+\) 与 \(q^-\) 除以同一个 \(m=\|x\|_1\)；分别归一会改变 \(\rho\)
2. 标准化统计量只来自 fit；重跑 test 不改 mean/std
3. \(\lambda\) 只在 calibration 选，test 重跑 \(\lambda\) 不变
4. P3 非对角严格为 0
5. \(\log|\Sigma|\) 进入 \(s_k\)；关掉它分数会变
6. 均等先验：计数不同的两个单元，先验项相同
7. 样本不足的单元走 pooled，不炸奇异
8. P0 与既有 J0 / L1 源在约定容差内对齐（若输入一致）

---

## 7. 第一刀明确不做

- 在线更新 \(N,\sum z,\sum zz^\top\)（过门后才排队，本文不实现）
- sharpness / spread 写进 \(z\)（已知与 COM+P2P 几乎重复；若要做，只能当更新权，而本刀无更新）
- HDC / 超向量
- 量化、CAM、行宽、读取倍数当目标
- GMM、NCA、MLP、局部 LDA 投影（P2b 只用 \(\lambda=1\)）
- shift-min
- 确认集
- 改 JSSC/DAC tex 摘要数字

---

## 8. 交回什么

1. 可运行的 `ro_ss_qda` + pytest
2. 四条开发记录上 P0–P3/P2b 总表（accuracy、同 home p10、最弱单元、\(\lambda\)）
3. `docs/ro_ss_qda_results.md`：过门与否、按第 5 节一句结论
4. D1 表或散点数据（不必挡结论）
5. provenance / sha256
6. 不要提交大 npz，不要 commit 除非用户要求

优先顺序：描述子 + P1 对齐 → P0 复现 → P2/P3 → P2b → D1 → 写结果。
