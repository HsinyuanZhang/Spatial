# COM-P2P Spike Sorting 方案 Handoff（含 ShiftCAM 新方向）

> 本文档供多 AI agent / 协作者 brainstorm 使用。目标是统一上下文，聚焦下一步方案设计。
> 最后更新：2026-08-21
> 仓库：本仓库（Spatial）是 STAR-Mem spike sorting 的算法 + 实验主体。
> 数据集（`new_datasets/`）、论文工程（`DAC2027/`）位于同级父仓库 `SNN_SpikeSorting`。

---

## 0. TL;DR

我们在做**面向 CiM/CAM 硬件的 spike sorting**，当前方案是 **COM（质心）+ P2P（峰峰值 footprint）双门控两阶段分配**。该方案在 HJ 静态数据上 Oracle 0.968、跨场景 0.903，但存在三个未解决的硬伤：

1. **Home channel 跳动**：P2P footprint 的 slot 含义跨 spike 不一致 → CiM 难以直接映射
2. **Drift（漂移）全部失败**：static range、online EMA、multi-prototype 均未通过门槛
3. **高密度探针 unique-ID 失败已实测**（Yger 252-ch 2-D MEA、KS4 `sim_no_drift` 384-ch）：主因是每电极多个 unit，不是单细胞 drift。COM/main-channel WTA 不能当分类器；Yger 上 COM 半径门可做候选球。Joint 0.99-recall/4× gate 仍未过。

**最新方向**：借鉴 [ShiftCAM (ICCAD 2024)](https://dl.acm.org/doi/10.1145/3676536.3676800) 的 Shifted Hamming Distance 思想，将 P2P footprint 改为**相对 home 的物理偏移列**，并做 **shift-min 匹配**，统一解决 home 跳动 + drift + CiM 映射三个问题。

---

## 1. 任务目标与约束

### 1.1 核心任务

为高密度电极探针设计一套**可在 CiM/CAM 硬件上实现**的尖峰分类流水线（代号 **STAR-Mem**），面向 DAC2027 投稿。

```
检测到的 spike → Stage-1 空间特征门控（CiM/CAM，廉价）→ 候选 unit 集
             → Stage-2 波形精分（数字域，昂贵，仅对 ≤3 候选）
```

### 1.2 硬约束

| 约束 | 说明 |
|---|---|
| 探针兼容性 | **必须同时兼容线性探针与 2D 探针**（Neuropixels 交错排列、密排六边形等） |
| Drift 适配 | **必须处理 unit 中心漂移**（可达数个通道） |
| CiM 友好 | 列含义固定、无需额外存储 neighbor_ids、可量化到 4-5 bit |
| 低 bit | 空间描述符目标 D=9（2 COM + 7 P2P），5-bit 量化 → 45 bits/spike |
| 无 GT | 在线部署不依赖测试集/ground truth 标签 |

### 1.3 验收门槛（6 条，当前无配置全部通过）

1. HJ + MEArec 候选召回率 ≥ 99%
2. 波形模板读取量 ≤ 全扫描的 1/4（≥4× 加速）
3. 紧凑数字波形表示与 64-sample teacher 差距 ≤ 1 pp
4. 能耗/面积优于 indexed-SRAM 基线
5. 需在真实高密度 2D 探针上验证
6. 在线部署无 GT 依赖

---

## 2. 数据集

### 2.1 Hybrid Janelia (HJ) — 主力 16 通道

| 属性 | 值 |
|---|---|
| 探针 | siprobe，16 通道线性，20 µm 间距 |
| 采样率 | 30 kHz |
| 场景 | 6 static + 6 drift，各 600s |
| 单元数 | static 10–23；drift 11–22 |
| 尖峰数 | static 28k–74k；drift 34k–71k |
| SNR 门槛 | T=8 |
| 路径 | `../new_datasets/hybridjanelia/*_16c_600s_*_filtered_gt.npz` |

NPZ schema：`recording (C,S)`, `geom (C,2)`, `event_times`, `event_labels`, `event_main_channels`, `unit_main_channels`, `unit_snrs`。

**drift 场景系统性更难**：soft_loc drift 召回 0.920 vs static 0.950。

### 2.2 MEArec v1 — 合成 32 通道 2D

| 属性 | 值 |
|---|---|
| 探针 | Neuronexus-32（2D shank，3×~11 交错排列，**非 32×32 网格**） |
| 变量 | units ∈ {10, 20} × SNR ∈ {2,3,5,8,12} × drift ∈ {none, mild} × seeds |
| 文件数 | 20 NPZ，120s/文件，30 kHz |

**注意**：20 个文件是 4 种时间配置 × 5 种 SNR 渲染，**不是独立生物学重复**，不得当作独立证据汇总。

### 2.3 Yger 252-ch 2-D MEA（Zenodo 1205233，已进 DAC packing 节）

| 属性 | 值 |
|---|---|
| 探针 | MCS 16×16，30 µm，252 extra 通道，20 kHz |
| 生物学 GT | **每段 1 个 juxta 细胞**（文献也只给这个细胞打 accuracy） |
| Spatial 伪 GT | KS4 Th=13：362 cluster / 314 `good`；60 s train 358 |
| 位置抖动 | KS4 `spike_positions` 中位 RMS **3.5 µm**（< 间距） |
| 60 s 事件 | 60 600 |
| Home | 全阵列 trough 21% 落在 90 µm 外（重叠偷 home，不是 drift） |
| 3-pitch unique-ID | main-channel 指数 10.8%；home (x,y) 45.6%；COM float **56.0%** |
| COM 半径门 | R=70 µm 召回 0.990、均 29/358 类、12.4×；名单内 WTA 仍 ~55% |
| Oversplit | 启发式 14 对可合并；无时间对半互斥；不是「314→200」 |
| HDD | `/mnt/data/.../zenodo_1205233_*`（禁止拷到 SSD） |

文献打分方式：Yger 2018 / SpikeForest paired-64ch / SpikeInterface 8/19 筛选，都是 **juxta 那一个细胞** 的 coincidence，不是 300 个 cluster 的普查。

细节：`docs/yger_ks4_pseudo_gt.md`，`docs/yger_com_direct_assign.md`。DAC 账本 C18–C20、C22。

### 2.4 KS4-paper `sim_no_drift`（Figshare 25298815，已进 DAC packing 节）

| 属性 | 值 |
|---|---|
| 探针 | Neuropixels 384 AP，30 kHz，纵间距 20 µm |
| GT | 模拟器真标签，**1200 unit**，全长 45 min |
| 60 s | 428 261 事件，1200 个 unit 都开火 |
| Occupancy | 338/384 电极占用，**3.55 unit/占用电极**（最多 40）；93.6% 共用电极 |
| 拥挤上限 | 完美 majority-home 查找 **56.4%** unique-ID |
| 3-pitch main-channel | **19.1%**（oracle majority home 28.0%） |
| COM float | **17.0%**（相对 1/1200 约 204×，与 Yger COM 的「相对随机」同量级） |
| 远端偷 home | 3-pitch 后仅 **1.3%**；全局 trough **60.7%**（中位 421 µm → 分类 0.48%） |
| HDD | zip 内 cbin 流式读前 60 s，不拷 SSD |

**19% 的主因是同电极拥挤，不是远处 spike。** 关掉 3-pitch 才会变成远端盗窃。

细节：`docs/ks4sim_no_drift_com.md`。DAC 账本 C21。

### 2.5 其他（次要）

- 2D Grid 1024ch（32×32 honeycomb）— legacy，未用新管线重跑
- CortexLab NP（128ch）— pending re-run
- Quiroga 单通道 — 用于 SpikingJelly SNN 管线
- Figshare `sim_fast_drift` 等 — 2026-08-20 仍在 HDD 下载（约 87%）

---

## 3. 当前 COM-P2P 方案

### 3.1 流程

```
原始信号 → 带通滤波 (300–6000 Hz) → 尖峰检测 → 确定 home channel
  → build_knn_table_with_self(geom, K)   # K 近邻，slot 0 = 自身
  → extract_local_p2p                    # 每邻居通道 P2P = max(seg) - min(seg)
       ├─ com_features()                 # P2P 加权几何质心 → 2D (cx, cy)
       └─ footprint_p2p_features()       # max 归一化的 P2P 向量 → K 维
  → Stage-1: COM(L2) ∧ P2P(L1/L2) 双门控 → 候选 unit
  → Stage-2: 候选中波形最近质心
  → Fallback: 归一化 COM+P2P 综合打分
```

### 3.2 关键代码位置（相对本仓库根）

| 模块 | 路径 |
|---|---|
| 特征核心 | `algorithms/spatial_footprint.py` |
| 两阶段门控 | `algorithms/two_stage_assign.py` |
| 无监督管线 | `algorithms/unsupervised_two_stage.py` |
| 特征构建入口 | `experiments/spatial_cim_common.py` |
| 在线 drift（失败） | `algorithms/online_spatial_adaptation.py` |
| 因果滤波（未集成） | `algorithms/causal_preprocessing.py` |

### 3.3 实验结果

**HJ 16ch scene 11，K=7：**

| 特征 | 维度 | Oracle | SOM |
|---|---|---|---|
| soft_loc (COM+spread+sharpness) | 4 | **0.916** | **0.833** |
| footprint_p2p | 7 | 0.806 | 0.642 |

**两阶段（HJ 跨 12 场景均值）：**

| 方法 | Oracle |
|---|---|
| com_and_p2p | 0.871±0.034 |
| **two_stage_waveform** | **0.903±0.038** |
| unsup_two_stage_waveform | 0.812±0.043 |

Fallback rate 2.7%，mean candidates/spike 1.21。

### 3.4 量化验证（CiM 近似）

| 编码 | 与 float 差距 |
|---|---|
| L1 | +0.1 pp（无损） |
| 4-bit code | −0.1 pp |
| Hamming 2-bit | **−2.9 pp（悬崖）** |

实用配方：max-normalize → 4-bit → L1，COM 保持低 bit 比较。

### 3.5 失败分析

- soft_loc 错误 64.1% 是 **same-channel collision**（同 home + Δcy≈0）
- soft_loc + Peak-FSDE 将 oracle 91.6%→93.0%，~61% 错误被波形挽回

---

## 4. 三个未解决的核心问题

### 4.1 Home Channel 跳动（CiM 映射障碍）

**问题**：KNN 表按距离排名，slot 0 永远是 home。同一神经元 home 在 ch12/ch13 间跳动时：

```
home=12 → slots: [ch12, ch11, ch13, ch10, ch14, ch9]
home=13 → slots: [ch13, ch12, ch14, ch11, ch15, ch10]
```

- **slot 含义跨 spike 不一致** → CAM 阵列第 j 列对不同 spike 代表不同物理通道
- **同一神经元零噪声 L1 距离 = 1.8**（纯 slot 错位，非真实差异）
- **central_ratio 更严重**：分母（slot 0 = home P2P）也变了
- COM 不受影响（用真实几何坐标加权质心）

**影响**：要匹配就得额外存 `neighbor_ids` 做软件重排，消解了 CiM 的核心优势（列并行、无数据搬移）。

### 4.2 Drift 全部失败

| 尝试 | 结果 |
|---|---|
| Frozen static ranges | drift 0.813 vs static 0.855 |
| Online EMA integer centroids | **−3.0 pp**（比 frozen 更差，错误标签正反馈） |
| Oracle-refit upper bound | 仅恢复 4.3 pp（目标 7.9 pp） |
| First-detect routing | drift 上 −4.6~−5.7 pp |
| Multi-prototype P≤4 | HJ 0.898（需 P=4），MEArec 无 4× 点 |

**结论**：没有任何 drift 方案通过门槛。根因：静态 per-unit 范围无法表示连续漂移；自预测标签在线更新有害。

### 4.3 高密度 2D 探针未验证

- MEArec 32ch 是唯一 2D 数据，且 20-unit 密集场景是瓶颈（com_and_p2p 0.791）
- 1024ch 2D grid 未用新管线重跑

---

## 5. ShiftCAM 启发的新方案

### 5.1 ShiftCAM 核心可迁移思想

[ShiftCAM (ICCAD 2024)](https://dl.acm.org/doi/10.1145/3676536.3676800) 原本解决**基因组测序的 indel 错误**，核心工具是 **Shifted Hamming Distance (SHD)**：对查询序列做多个偏移量的 Hamming 距离，取最小值，近似 edit distance。

**数学同构**：indel 导致序列平移 ≡ home 跳动导致 P2P footprint 平移。SHD 的"多偏移取最小"可直接迁移。

> 参考：[ShiftCAM IEEE 版本](https://ieeexplore.ieee.org/document/11126302/)

### 5.2 方案：Relative-Offset Footprint + Shift-Min Matching

#### Step 1：规范化 slot 顺序（解决"列含义固定"）

抛弃 KNN 距离排名，改用**相对 home 的物理偏移**作为列索引：

```
线性探针：列 = [home-3, home-2, home-1, home, home+1, home+2, home+3]

2D 探针：列 = patch_grid 中固定的 (Δr, Δθ) 偏移
  十字形示例：(0,0), (-1,0), (+1,0), (0,-1), (0,+1), (-1,-1), (+1,+1)...
```

- 列含义跨 spike 一致，**无需 neighbor_ids**
- home 偏移导致整组通道平移，但 (Δr, Δθ) 不变 → **天然不变性**
- 需为每种探针几何预定义 patch_grid 模板（线性 1D 7 点 / Neuropixels 交错 2D / 六边形 2D）

#### Step 2：Shift-Min 匹配（解决 home 跳动 + drift）

```
d(spike, template) = min over s ∈ [-S..S] of L1(spike, shift(template, s))
```

- 线性探针：S 个 1D 整数平移，S=2 覆盖 ±2 通道 drift
- 2D 探针：S 个 2D 平移（±1 在 r 和 ±1 在 θ），S=4~8
- **吸收 unit 中心 drift**：drift 使中心从 ch12 漂到 ch11 ≡ footprint 平移 1 位，shift-min 自动对齐

**预期良性循环**：shift-min 降低 intra-unit 距离 → τ_p2p 收紧 → 候选集更小 → Stage-2 更准。

#### Step 3：CiM/CAM 映射

| 选项 | 原理 | 优点 | 缺点 |
|---|---|---|---|
| **A（推荐起步）：多副本预 shift CAM** | 同一模板存 S+1 份，各预 shift 不同量，查询广播后并行匹配 + WTA | 阵列内部纯 CAM 无修改，验证成本最低 | 面积 ×(S+1) |
| B：行级 permutation 网络 | 阵列前加可配置置换，选 shift 后重排列 | 面积不增 | 延迟 ×S |
| C：时域 shift（最忠实 ShiftCAM） | P2P 编码为脉冲延迟，shift = 时钟延迟 | 无数据搬移 | 电路复杂 |

#### Step 4：训练阶段漂移感知模板

drift 是连续漂移，shift-min 的一个模板覆盖 ±S 通道范围。训练时对 unit 的所有 spike 在 COM 空间做 KNN 聚类，每个空间簇中心 spike 作为模板。簇数 = ceil(drift_range / S)。

#### Step 5：门控组合

- **Stage-1a（粗筛）**：纯 COM 距离门控（对 drift 鲁棒）→ 候选 unit 集
- **Stage-1b（CiM 精筛）**：在候选上跑 shift-min CAM 匹配 P2P footprint
- **Stage-2（数字域）**：对 ≤3 候选做波形 PCA 最近质心

### 5.3 与当前代码的差异

| 模块 | 当前 | 改后 |
|---|---|---|
| `build_knn_table_with_self` | 按距离排名 | 按 patch_grid 偏移 |
| `extract_local_p2p` 输出 | (n,K) + neighbor_ids | (n,K)，列含义固定，无 neighbor_ids |
| `stage1_candidates` | 单次 L1/L2 | shift-min L1（S 偏移取最小） |
| `calibrate_thresholds` | 95th percentile | 需重标定（intra-unit 距离降低） |
| `quantize_p2p` | per_spike_max | 不变 |

### 5.4 风险与开放问题

| 风险/问题 | 现状 |
|---|---|
| S 的选取 | S 太小挡不住 drift，太大引入 false match；建议 S=2 起步实测 |
| 2D shift 组合爆炸 | (2S+1)²=25 个 shift → 需先用 COM 筛到 ≤5 候选再跑 2D shift |
| patch_grid 在非密排探针有空位 | 填 0 起步（远端幅度小影响有限）或加权掩码（需 CAM 支持掩码位） |
| shift-min 是否真能通过 G5 drift 门槛 | **未验证**，这是最关键的实验 |
| 不同探针几何的 patch_grid 标准化 | 需为线性/交错/六边形分别设计模板 |
| shift-min 后 τ_p2p 重标定 | 预期收紧，需实测确认 |
| CiM 面积估算 | S+1 副本 vs permutation 网络的 trade-off 未量化 |

### 5.5 预期收益

1. **drift 问题首次有可行方案**（当前所有 drift 尝试均失败）
2. **CiM 映射干净**：列固定、紧凑 K 列、无需 neighbor_ids
3. **精度提升**：shift-min 消除 home 偏移导致的虚假距离
4. **门控收紧**：τ_p2p 可降，候选集更小，Stage-2 更准

---

## 6. 下一步建议（优先级排序）

1. **实现 patch_grid 生成器**：支持线性 1D、Neuropixels 交错 2D、六边形密排三种模板
2. **实现 shift-min 匹配函数**：在 `algorithms/two_stage_assign.py` 中替代 `stage1_candidates`
3. **在 HJ drift 场景上验证**：对比 frozen / online-EMA / shift-min，目标通过 G5 门槛
4. **标定新 τ_p2p**：验证门控收紧的良性循环假设
5. **2D 探针验证**：在 MEArec 32ch 上验证 patch_grid 和 2D shift
6. **CiM 面积估算**：S+1 副本 vs permutation 网络的面积/延迟 trade-off

---

## 7. 关键文件索引（相对本仓库根）

| 类别 | 路径 |
|---|---|
| 特征核心 | `algorithms/spatial_footprint.py` |
| 两阶段门控 | `algorithms/two_stage_assign.py` |
| 无监督管线 | `algorithms/unsupervised_two_stage.py` |
| 特征构建入口 | `experiments/spatial_cim_common.py` |
| 在线 drift（失败） | `algorithms/online_spatial_adaptation.py` |
| 因果滤波（未集成） | `algorithms/causal_preprocessing.py` |
| 架构文档 | `docs/dac_two_level_memory_pipeline.md` |
| 硬件映射规范 | `docs/hardware_mapping_spec.md` |
| 结果汇总 | `../DAC2027/data/current_results.tex`（父仓库） |
| 证据台账 | `../DAC2027/notes/EVIDENCE_LEDGER.md`（父仓库） |
| 论文项目 | `../DAC2027/`（父仓库，ACM sigconf, Overleaf） |
| 数据集目录 | `../new_datasets/hybridjanelia/catalog.json`（父仓库） |

---

## 参考文献

- [ShiftCAM: A Time-Domain Content Addressable Memory (ACM ICCAD 2024)](https://dl.acm.org/doi/10.1145/3676536.3676800)
- [ShiftCAM (IEEE Xplore)](https://ieeexplore.ieee.org/document/11126302/)
