# EMST 时空事件型 Event-KG 项目书

## 1. 项目背景与动机
电磁态势认知面临“事件—观测—地理/频谱”多源异构信息融合与补全问题。传统方法往往依赖外部数据或静态图结构，难以处理时间依赖与观测噪声。本项目目标是在**可复现的合成环境**中，从零构建时空事件型 Event‑KG 数据集，并以 PyTorch Geometric 实现**异构时空 GNN/Transformer**，完成事件到地理单元与频段的补全，同时提供**不确定性估计与校准**，形成可论文化的实验体系。

## 2. 项目目标
1. 构建可复现的 Event‑KG 合成数据集，保证时间切分无泄漏。  
2. 实现异构时空 GNN/Transformer 模型：Event→GeoCell、Event→Band 两个补全任务。  
3. 提供 MC Dropout 与 Temperature Scaling 校准评估。  
4. 完成可解释的消融实验与显著性检验，形成论文级证据链。

## 3. 研究问题与范围
### 3.1 核心问题
- 时空事件在异构图上如何被有效建模与补全？
- 时序边与 Δt 是否带来真实增益（time‑split 下）？
- 观测信息强弱如何影响 Band 与 Geo 两任务的可学性？
- 校准与不确定性评估是否带来可信度增益？

### 3.2 范围与约束
- **仅使用合成数据**，不依赖外部数据源。  
- **严格 time‑split**，禁止未来信息泄漏。  
- 模型使用 PyG `HeteroData + HeteroConv`。  
- 输入边不包含 `occurs_in/overlaps_band`（避免答案泄漏）。  

## 4. 数据集设计
### 4.1 合成空间与时间
- 空间网格：`Nx=20, Ny=20`（400 geocell）  
- 频段桶：`B=32`  
- 传感器：`S=20`  
- 时间步：`T=30000`（默认）  

### 4.2 真值事件生成
每个时间步采样若干真实事件，属性包含：  
- `t_start, t_end`（持续时间 5~50）  
- `geocell_id, band_id_true`  
- `power, bw`（正值噪声）  

### 4.3 观测生成（局部可观测）
真事件只在半径 `R` 内被少量传感器观测，避免“全传感器复制”带来的噪声结构。  
- 观测概率：`p_detect = (1 - p_fn) * sigmoid(a - b*dist) * reliability`  
- 漏检/误报：`p_fn_base=0.15, p_fp_base=0.004`  
- Band 漂移：默认 ±1 桶  
- 置信度：真事件 `Beta(8,2)`、假事件 `Beta(2,8)`  

### 4.4 数据产物
输出包含标准 KGC 三元组与属性表：  
- `train/valid/test` 三元组 + `entity2id/relation2id`  
- `event/sensor/geocell/band` 属性表  
- `pyg.pt` 异构图文件  

## 5. 图构建与学习任务
### 5.1 节点与特征
- `event.x`：`[t_norm, band_obs_norm, bw, power, conf]`  
- `sensor.x`：`[reliability]`  
- `geocell.x`：`[x_norm, y_norm]`  
- `band.x`：`[f_center_norm]`  

### 5.2 边类型（输入）
- `(event)-[observed_by]->(sensor)`  
- `(sensor)-[located_in]->(geocell)`  
- `(event)-[prev_event]->(event)`，仅连向过去，edge_attr=Δt  

### 5.3 监督标签
- `y_geo = geocell_id_true`  
- `y_band = band_id_true`  
仅对真事件计入 loss/指标；FP 标签为 `-1`，mask 过滤。  

## 6. 模型与训练
### 6.1 模型结构
- 2 层 `HeteroConv`  
- `prev_event`: `TransformerConv`（edge_dim=Δt 编码）  
- `observed_by/located_in`: `SAGEConv`  
- 输出：`z_event` → `geo_head` / `band_head`  

### 6.2 训练策略
- time‑split：train/valid/test = 60/20/20  
- mini‑batch：`NeighborLoader`  
- loss：`CE_geo + CE_band`  
- 所有随机性固定 seed，可复现  

## 7. 评估与校准
### 7.1 排名指标（deterministic）
- Accuracy / MRR / Hits@K（K=1,3,10）  
- 排名仅基于全类别 logits  

### 7.2 校准与不确定性（MC Dropout）
- Temperature Scaling（valid 上最小化 NLL）  
- ECE / Brier / NLL / Risk‑Coverage  
- 排名指标与校准指标分离报告  

## 8. 实验设计与消融
### 8.1 时空结构消融（核心证据）
- A0：无 `prev_event`  
- A1：有 `prev_event`，Δt=0  
- A2：有 `prev_event` + Δt  
预期：A2 ≥ A1 ≥ A0（Geo 的 MRR/Hits@10）  

### 8.2 Band 饱和解释（B1）
遮蔽 `band_obs` 特征，仅对 Band 任务两次 forward：  
- Geo：完整特征  
- Band：band_obs 置 0  
用于回答“Band 饱和是否来自观测泄漏”。  

### 8.3 稳定性与显著性
5 seeds 均值±方差；对 A1/A2 vs A0 做 paired bootstrap / t‑test。  

## 9. 复现流程
```
python data_gen/build_kg_files.py --config configs/default.yaml --out data/synth1
python pyg_data/build_heterodata.py --data_dir data/synth1 --out data/synth1/pyg.pt
python train.py --data data/synth1/pyg.pt --save checkpoints/model.pt
python eval.py --data data/synth1/pyg.pt --ckpt checkpoints/model.pt
python calibrate.py --data data/synth1/pyg.pt --ckpt checkpoints/model.pt --out checkpoints/calib.json
python eval.py --data data/synth1/pyg.pt --ckpt checkpoints/model.pt --calib checkpoints/calib.json --mc_dropout 20
```

## 10. 交付物
- 合成数据生成脚本与 KGC 文件  
- PyG 异构图构建脚本  
- 训练/评估/校准/消融脚本  
- 实验结果与显著性统计  

## 11. 风险与对策
- **任务过易（Band 饱和）** → 使用 B1 解释或提高漂移难度  
- **时序增益不稳** → 提升样本规模、引入更强时序信号  
- **time‑split 泄漏** → split 内建边 + sanity_check 断言  

## 12. 里程碑
1. 数据生成与图构建完成  
2. 模型训练与基础指标完成  
3. 消融与显著性验证完成  
4. 校准与不确定性评估完成  
5. 论文材料整理与交付  
