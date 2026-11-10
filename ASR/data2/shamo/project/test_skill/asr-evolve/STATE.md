# Loop State — ASR-Evolve

Last run: 2026-07-10（slim_chain_v1, CER=13.60%, 召回=79.67%）

## 高优先级（环路正在处理 / 等待人工）

- CER从14.52%降至13.60%（-0.92%），精简增强链有效
- 距离FINE阶段(<10%)还差3.6%，仍在COARSE阶段
- 今日训练配额 3/3 已用完
- 下一步：可继续在精简链基础上调参（如进一步调SNR/volume范围）

## 观察列表

- 召回率微降80%→79.67%（-1条），基本持平
- 精简链去掉clipping+noise后模型不再"过度保守"
- augment run CLI有bug（远程成功但本地误报失败），需手动merge或修复

## 当前最优

- 模型: slim_chain_v1 (model.pt.best) [node-6]
- CER: 13.60%
- Stripped CER: 14.95%
- 召回率: 79.67% (239/300)
- 测试集: v0.6.0 (300 samples)
