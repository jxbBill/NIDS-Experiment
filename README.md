# Trusted Pseudo-label Continual NIDS Prototype

本目录是第 3 节研究方案的代码化原型。


## 运行环境

建议在 Linux 服务器上建立虚拟环境：

```bash
cd 论文/MyCode
pip install -r requirements.txt
```

## 快速运行

完整默认实验：

```bash
python train.py --dataset nsl
```

快速冒烟测试：

```bash
python train.py \
  --dataset nsl \
  --max-train-samples 2000 \
  --max-test-samples 800 \
  --initial-epochs 1 \
  --online-epochs 1 \
  --window-size 400 \
  --max-windows 3 \
  --tau 0.5 \
  --consistency-tau 0.0 \
  --always-update
```

也可以直接运行：

```bash
bash scripts/run_nsl.sh
```

## 输出文件

默认输出到 `outputs/nsl_时间戳/`：

- `config.json`: 本次实验参数。
- `window_log.csv`: 每个在线窗口的漂移、伪标签数量、窗口指标和记忆缓冲区状态。
- `final_metrics.json`: 测试集最终指标。
- `model.pt`: 最终模型参数。

## 核心流程

1. 用少量初始标注样本训练 AE+分类头。
2. 将剩余训练样本模拟为持续到达的无标签网络流量。
3. 对每个窗口输出攻击概率、重构误差、置信度和一致性分数。
4. 用 KS 检验比较当前窗口与历史参考分布，判断是否发生漂移。
5. 只把满足 `confidence >= tau` 且 `consistency >= consistency_tau` 的样本作为可信伪标签。
6. 将可信样本写入类别平衡记忆缓冲区，并与 replay 样本一起更新模型。
7. 用 held-out 测试集评估最终检测性能。
