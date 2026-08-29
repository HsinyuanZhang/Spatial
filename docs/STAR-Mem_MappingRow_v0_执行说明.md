# STAR-Mem Mapping Row v0 — Grok Build 执行说明

面向仓库 [HsinyuanZhang/Spatial](https://github.com/HsinyuanZhang/Spatial)。
工作名 STAR-Mem。DAC 稿暂定标题：A Spatial Associative Memory Pipeline for Large-Scale Extracellular Spike Sorting。

本文是给 **Grok Build** 的执行规格，不是论文草稿。先做静态证伪，不要先做在线、不要先做 CAM/RTL。

---

## 0. 一句话任务

实现并跑通一组冻结对照，回答：

> 在已经归一化的空间形状上，**逐单元精度 / 区间违例映射** 是否比 **平均峰峰值 + 整数 L1 半径** 更能拉开「真单元 vs 最近干扰单元」，并在 **候选召回 ≥ 0.99** 时把波形模板读取降到 **≥ 4×**。

过门才谈在线更新和电路。不过门就停，不要扫分位、掩码数、权级数、原型数。

---

## 1. 仓库边界（必须遵守）

- 只改 `HsinyuanZhang/Spatial`。不要碰 `SpikeSortingSNN`、`iBCI`。
- 现有合取门仍有效，见 `docs/dac_two_level_memory_pipeline.md`：
  1. HJ 与 MEArec 两边 held-out 真单元候选召回 ≥ 0.99
  2. 全局波形模板读取 ≥ 4× 减少（相对全单元 64×5 SAD）
  3. 不靠 Range-CAM / 空间唯一命中早退完成流量
  4. 量化、阈值、区间、精度 **只在 fit/calibration 上估计**；test 标签只用于指标
- 现有否定分支 **禁止重开**：静态 1/2/4 权重扫、二进制 mask 数扫、`P≤4` 多原型、唯一命中早退、内外级联认证早退、共享 dyadic 时间素描当主线。
- 当前检索实验默认：真值事件、真值行、整段零相位滤波。第一刀保持这个 isolation，诚实写成 offline component isolation。
- 一级是检索器不是分类器。冻结 5-bit L1 p99.9 源上，四个开发记录约 4999 个 test 事件里约 4996 个 `C>1`。不要把空间 argmin 当成最终标签。
-  dense 探针（Yger 252ch、`sim_no_drift` 384ch/1200 units）上 COM / 主通道不是 unique ID。粗选只出候选球。
- 目录 / tile / 质心球 **不能输出单元标签**。
- 第一刀硬件叙事只实现 **indexed SRAM + 数字比较**。不要写 RRAM/CAM 宏、refresh=learning、PPA。

开发集（先跑、冻结配置）：与仓库已锁定的四条 60s 记录相同（两条 HJ、两条 MEArec）。通过后再跑剩余 HJ/MEArec 和 dense 探针，**不再改配置**。

时间划分（冻结）：

- 前 50% fit
- 中间 25% calibration
- 最后 25% test
- 相同时间戳必须留在同一段，且仍是独立事件

---

## 2. 要存的对象（三家汇总后的最小对象）

记忆行 **不存**「这个单元平均长什么样」的平均 P2P 向量。
记忆行 **存** 一个小映射 \((\mu, \Lambda)\)（或它的区间读法），查询只比形状。

### 2.1 事件侧（查询包）

对触发通道邻域 \(K=7\)：

1. **POSNEG 形状，不要先合成 P2P。**
   - \(q_{\mathrm{pos}}[7]\)：各通道正向峰值
   - \(q_{\mathrm{neg}}[7]\)：各通道负向谷值绝对值
   - 两个单元可以 P2P 接近，但一个以负谷为主、一个有明显反弹正峰。合成 P2P 后不可恢复。
2. **质心** \(q_x, q_y\)：几何加权 COM，已有 `soft_loc` / COM 实现可复用。
3. **尺度 \(g\)**（4-bit）：局部 7 通道最大绝对幅值（或正负幅值和）的低比特码。
   - \(g\) **不是身份**。只用于把整体增益与形状分开，并产生 `scale_alert`。
   - 形状归一化：\(s_i = \mathrm{round}(15 \cdot a_i / \max(g, 1))\)。硬件可用 16 项倒数 LUT + 移位，**不要通用除法器**。
4. **quality flags（3 bit，只走控制路径）**
   - `quality_sat`：锁存/量化饱和
   - `quality_cut`：窗没盖住峰或谷
   - `quality_overlap`：疑似重叠 / 异常多通道同时激活
   - 这些位 **不得** 提高某维匹配权重。只能禁止更新，或强制进时间级。

建议事件包：66-bit shape key（COM 10 + pos 28 + neg 28，按 5-bit COM / 4-bit POSNEG 记账）+ 4-bit \(g\) + 3-bit quality。第一刀若现有流水线仍是 5-bit COM+P2P，允许先在软件 float 上算 POSNEG 再量化；对照里必须有「只改输入、不改关系」的一臂。

### 2.2 模板侧（映射，不是均值）

每个单元一行，最小充分统计：

- 形状中心 \(\mu\)（对归一化 POSNEG + COM）
- 对角精度 \(\Lambda = \mathrm{diag}(\sigma^{-2})\)，来自 **该单元 fit 历史**，不是当前事件哪儿大
- 可选：由 \((\mu, \sigma)\) 读出的每维区间 \([\mu - \kappa\sigma, \mu + \kappa\sigma]\)
- 尺度允许范围 \([g_{\mathrm{lo}}, g_{\mathrm{hi}}]\)，只产生 `scale_alert`，**第一刀不因尺度越界删候选**

匹配语义（选一个主实现，另一个当读法）：

**主实现（推荐，webi）：加权违例 / 加权 L1，不是离均值的普通 L1。**

对第 \(i\) 维：

- 若采用区间读法：\(\mathrm{raw\_violation}_i = \max(0, \ell_i - q_i, q_i - h_i)\)。落在区间内贡献 0，**不再比离中点多近**。
- 再按模板精度归一：\(\mathrm{penalty}_i = \min(3, \lceil \mathrm{raw\_violation}_i / 2^{p_i} \rceil)\)，其中 \(p_i \in \{0,1,2,3\}\) 由 **区间宽度** 推出，不是另学一套 1/2/4 权重。
- \(S_u = \sum_i \mathrm{penalty}_i\)。16 维每维最大 3，\(S_u \in [0, 48]\)，6-bit 累加器够。

**等价对象（Claude / webi）：** \(d_\Lambda(q,\mu) = (q-\mu)^\top \Lambda (q-\mu)\) 或加权 L1 \(\sum_i |q_i-\mu_i|/\sigma_i\)。区间半宽 \(\propto \sigma_i \propto 1/w_i\)。不要同时实现两套再挑好看的；软件里用加权违例，文档里写明它是 \(\Lambda\) 的 L∞/饱和读法。

**删失（Claude，必须实现，不要做成硬 don't-care）：**

- 观测安静 **不是** 缺失。
- 模板该维 \(\mu\) 大、观测安静：强不匹配（高 penalty）。
- 模板该维贴近地板、观测安静：匹配（penalty 0 或单边区间下界饱和）。
- 禁止把低幅度维直接 mask 掉。这是 MEArec 掩码实验失败的原因：不要-care 制造无限公差，密集单元碰撞。

**三源规则（Claude）：**

| 源 | 是什么 | 放哪 |
|----|--------|------|
| 传感器侧 \(\sigma_{n,c}\) | 通道噪声地板 / 阻抗，与类无关 | 前端白化或特征缩放，**不进 CAM 行** |
| 单元侧 \(\sigma_{d,u,c}\) | 该 unit 该维历史稳定性 | **写进行**，即 \(\Lambda\) |
| 观测侧 | 本次 SNR / 饱和 / 重叠 | **只走控制路径**：冷启动、弃权、禁止更新 |

观测侧权不得进入 \(S_u\)。

### 2.3 不要做成 ChatGPT 满配 192-bit 行

ChatGPT 的 192-bit 行 + 128-bit update SRAM + novelty FIFO + 合并状态机，**第一刀不做**。

从那份清单 **只吸收**：

- POSNEG 拆峰谷
- \(g\) 与 shape 分离，`scale_alert` 不删候选
- quality 只控更新
- 区间内部不比离中心多近
- `tau_candidate` 与 `tau_update` 分离（更新更紧）——第一刀静态实验只用 `tau_candidate`
- 实验纪律：固定 5/95 与 99.9，不扫分位
- 第一版硬件是 indexed SRAM + 数字比较树，不是纯 Range-CAM（Range-CAM 只有 AND 命中位，产不出软违例分数）

**不吸收：** 192-bit 为目标行宽、16 套独立盒子再叠 16 个与宽度重复的 precision 字段当新信息、第一刀就做在线平移/SUSPECT/开簇/合并、唯一命中早退、给同一 unit 加第二平均原型。

若实现需要一个具体行宽记账，静态主版本上限记为：

- COM 区间 20 bit（4×5）
- 14 个 POSNEG 区间 112 bit（14×8）或改存 \(\mu+\sigma\) 更短
- 精度若由宽度推出则 **不要再单独立 32 bit**
- \(g\) 范围 8 bit
- `tau_candidate` 6 bit

宁可先 80–140 bit 验证对象，也不要为「完整」堆到 192。过门后再压。

---

## 3. 控制器（第一刀）

对每个事件：

1. 目录 / tile / 质心球得到 `active_row_mask`（弱门控，可先用现有 directory；dense 探针用 COM 球作诊断，不替代 HJ/MEArec 合取门）。
2. 对活动行算 \(S_u\)、`scale_alert_u`。
3. \(u \in C \iff S_u \le \tau^{(u)}_{\mathrm{candidate}}\)。
4. 保留：\(C\)、最小分 \(S_1\)、次小分 \(S_2\)、`top1_id`、`top2_id`。
5. **禁止** \(C=1 \Rightarrow\) 直接当标签。第一刀所有歧义与非歧义事件都继续记 Level-2 读取计数：凡是要报「波形读取倍数」，分母仍是「若 \(|C|>1\) 则读 \(|C|\) 条时间模板」。可以 **额外统计**「强内部 margin」事件比例（`top1` 成熟、\(S_1\) 很小、\(S_2-S_1 \ge 4\)、无 scale_alert、quality 全 0），但 **不要早退**。

`tau_candidate`（calibration，冻结）：

- 对该单元 calibration 事件算 \(S_u\)
- \(\tau = \lceil\) 该单元 99.9 分位 \(\rceil\)
- 只允许这一个目标，禁止再扫半径/分位

`tau_update` 第一刀不算进门控，只预留字段或注释。

---

## 4. 必须跑的静态对照（冻结，按这个顺序）

在 **完全同一套事件、同一 split、同一 directory/粗选** 下比较。不要中途改 K、窗、滤波。

| ID | 名字 | 查询 | 行 / 关系 | 目的 |
|----|------|------|-----------|------|
| J0 | 现有基线 | 5-bit COM + P2P（现有 \(D=9\)） | 平均向量 + 整数 L1 + 逐单元 99.9 半径 | 仓库现状 |
| J1 | 只改输入 | COM + POSNEG（4 或 5 bit） | 仍是平均向量 + L1 + 99.9 半径 | POSNEG 本身有多少贡献 |
| J2 | 区间映射，无尺度分离 | 原始 POSNEG（未按 \(g\) 归一） | 区间违例映射（5/95 盒 + 宽度导出移位） | 「区间关系」本身 |
| J3 | 主版本 | \(g\)-归一化 POSNEG shape + COM，\(g\) 只报警 | 同上区间违例 / 加权违例 | 形状与尺度分开 |
| J4 | 去掉精度移位 | 同 J3 | 所有 \(p_i=0\)（纯区间违例，不按宽度归一） | 异方差归一是否有用 |
| J5 | 删失非对称（Claude） | 同 J3 | 模板大声+观测安静 = 高罚；模板近地板+观测安静 = 0 | 是否优于对称盒子 |
| J6 | 对数形状（可选，若 J3 周内有余力） | \(\log(a+\varepsilon)\) 减峰通道 | 平均或区间 | 增益变加法是否更稳 |

**不要加：** 不同 mask 数、不同 1/2/4 权重组合、不同 \(P\)、唯一命中早退、多套半径、pair-tap、dyadic 主线。

初始化（冻结，ChatGPT 纪律）：

- `low_i` / `high_i`：fit 的 5% / 95% 分位。不要同时扫 1/2/5/10。
- \(p_i = \mathrm{clamp}(\lceil\log_2(\mathrm{width}_i)\rceil - 1, 0, 3)\)，HJ 与 MEArec **同一规则**。
- \(g_{\mathrm{lo}}, g_{\mathrm{hi}}\)：fit 的 1% / 99%，只产生 alert。

---

## 5. 每个版本必须报告的指标

对 HJ/MEArec 每个 recording，以及合计（写明 recording-unweighted vs event-weighted）：

- 真单元候选召回（整体 **和** 每单元）
- 平均 \(|C|\)、p95 \(|C|\)
- \(C=0 / C=1 / C>1\) 比例
- **真单元 \(S\) 与最近干扰单元 \(S\) 的间隔**（均值、p10、p50）。这是「映射有没有成立」的主诊断。间隔不动 ≈ 换皮平均模板。
- 每维对违例 / 间隔的贡献（谁在拉开干扰）
- 波形读取倍数（按现有 `R_traffic` 定义，64×5、仅 \(|C|>1\) 计读）
- 每事件活动行数、比较次数（lower/upper compare、移位、饱和、累加）
- 行位宽（逻辑 payload，不含 ECC/ID）
- `scale_alert` 比例
- fit 前半 vs 后半的区间宽度稳定性

Dense 探针（J3 若在开发集出现联合通过点之后才跑，否则可只做诊断、不进合取门）：

- Yger 252ch：COM 球 + 映射后的召回与平均 \(|C|\)
- `sim_no_drift`：拥挤电极上的候选碰撞；主通道不得当 unique ID

合取通过（开发集，必须两边家族都过）才允许冻结配置去确认集：

1. HJ 与 MEArec 真单元候选召回都 ≥ 0.99
2. 波形读取 ≥ 4×
3. **不是** 靠 \(C=1\) 早退凑出来的
4. 没有单条低 SNR / 高单元数记录召回崩盘
5. 真/干扰分数间隔相对 J0 有可见拉开（在报告里写清分布，不要只报均值）

解释用：

- J1 > J0 且 J3 不行 → POSNEG 有用，区间关系不够 → **停映射行，不要扫分位**
- J3 > J2 → 尺度/形状分离是关键
- J4 ≈ J3 → 删掉精度移位字段
- J5 > J3 → 删失非对称写入论文对象
- J3 在 0.99 召回时读取仍约 1.2× → **停止这条映射关系**（ChatGPT 的停则，保留）

---

## 6. 实现落点（Spatial 仓库）

建议新文件，不要把否定实验改写成「好像还活着」：

- `algorithms/mapping_row.py`：事件包、归一化、区间/`penalty`、`S_u`、删失规则、`scale_alert`
- `algorithms/mapping_init.py`：fit 分位、calibration \(\tau\)、宽度→\(p_i\)
- `experiments/run_mapping_row_static.py`：J0–J5 一条命令跑完四个开发记录
- `docs/mapping_row_v0_plan.md`：把本文件的冻结字段抄过去（plan）
- `docs/mapping_row_v0_results.md`：跑完写结果 + 是否过门 + 停/进确认
- `docs/mapping_row_v0_provenance.md` + sha256：输入四文件、源文件、输出 CSV

必须复用、不要重写：

- 数据 loader、split、GT 事件对齐、collision-safe 时间戳
- 现有 J0 的 `adaptive_range_search` / 5-bit L1 作为基线对照
- `evaluation` 里已有的召回 / 读取记账；新指标（真/干扰间隔）另加，不要改旧 CSV 列语义
- 测试：非连续 unit ID、inclusive 边界、calibration 不碰 test、J0 与旧 runner 数值对齐（允许文档写明的舍入）

CLI 示例（实现时可微调，但要一条命令可复现）：

```bash
python -m Spatial.experiments.run_mapping_row_static \
  --pilot --duration 60 --quiet \
  --arms J0,J1,J2,J3,J4,J5
```

默认 `--pilot` = 锁定的四条开发记录。没有 `--confirm` 不得动确认集。

单元测试至少覆盖：

1. 区间内部 `raw_violation=0`，与到中点的距离无关
2. 模板 \(\mu\) 大、观测 0 → J5 高罚；模板近地板、观测 0 → J5 低罚
3. \(g\) 越界只立 `scale_alert`，不改变 \(S_u\)
4. quality 不进入 \(S_u\)
5. \(\tau\) 只来自 calibration，test 重跑 \(\tau\) 不变
6. J0 与现有 p99.9 L1 对照在同一输入上召回误差在约定容差内

---

## 7. 第一刀明确不做

- 在线平移 / 扩张 / SUSPECT / novelty FIFO / 开簇 / 合并
- 把 update counter 塞进搜索行
- GT 标签参与更新或选行
- 空间唯一命中早退
- 同一 unit 第二、第三个平均原型
- Range-CAM / RRAM 宏、综合、PVT、refresh=learning 当结果
- LLM / agent 进数据通路或论文贡献
- 把摘要里的 85–93% 空间可分、或 Yger 12.4× / sim 46.5× 当成已经通过的合取门
- 共享 dyadic、pair-conditioned taps 作为本轮主实验（时间分解是 **映射过门之后** 的第二轮）

---

## 8. 过门之后才排队（不要在本轮做，只写在结果文档末尾）

第二轮：时间表示分解（候选集合冻结，与现有 `docs/dac_two_level_memory_pipeline.md` 一致）

- float 教师 / 64×5 SAD / 16 样点 float 前缀 / 16 样点 5-bit 前缀 / 5-bit raw taps / float dyadic / 5-bit dyadic
- 目的是挑一个 **更新确认** 基线，不是本轮主线

第三轮：已知初始行、未知在线标签（Frozen / scale-only / shape-translation / shape+scale+时间一致 / Oracle 上界）

第四轮：无真值发现、COLD→MATURE、映射距离合并

硬件：算法过门后，indexed SRAM + 数字违例树 vs 旧 54-bit L1 vs 63-bit mask vs 74-bit weight。CAM 只当后续候选。

---

## 9. 论文对象怎么写（给结果文档用，不要先改 tex 大叙事）

可写进结果讨论的句子：

> 观测是非负空间场上的带精度点；行里存的是形状中心与对角精度（或其区间读法）；尺度单独报警；删失是单边信息，不是 don't-care。

不可写（本轮没有证据）：

- Range-CAM 早退
- 刷新即学习
- 空间已经可分、波形多余
- 已选出架构 / 已过合取门（除非数字真过）

若 J3/J5 不过门：结论写成「这条映射关系在当前 9–16 维低比特描述子上不能同时满足 0.99 召回与 4× 读取」，不要改分位再战。

---

## 10. 验收（Grok Build 交回什么）

1. 可运行的 `mapping_row` 实现 + pytest
2. 四个开发记录上 J0–J5 的一张总表（召回、\(|C|\)、读取倍数、真/干扰间隔）
3. `docs/mapping_row_v0_results.md`：过门与否、按第 5 节解释、下一步是确认集还是停
4. provenance / sha256
5. 不要提交大 npz、不要改 JSSC 论文 tex 的摘要数字

优先顺序：J0 对齐旧基线 → J1 → J3 → J4/J5 → 写结果。J6 可选。
