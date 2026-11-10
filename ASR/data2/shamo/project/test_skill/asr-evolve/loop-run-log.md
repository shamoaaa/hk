# Loop Run Log — ASR-Evolve

每次环路运行追加一条记录。

## 运行记录

<!-- 环路在下方追加 -->

{"timestamp":"2026-07-09T16:08:00Z","loop":"iterate","action":"baseline训练+评估","outcome":"baseline","cer_before":null,"cer_after":0.1555,"gpu_hours":0.1,"node_id":"node-1","details":"首轮baseline: epoch=5, lr=1e-4, batch=64, 无增强, CER=15.55%, 召回=77.67%"}

{"timestamp":"2026-07-09T16:35:00Z","loop":"iterate","action":"数据增强训练(完整链augment_factor=1)","outcome":"effective","cer_before":0.1555,"cer_after":0.1463,"gpu_hours":0.28,"node_id":"node-2","details":"CER 15.55%→14.63%(-0.92%), 插入Ins 30%→25%, 但删除Del 31%→39%升高。GPU2 OOM后切到GPU1成功"}

{"timestamp":"2026-07-09T17:20:00Z","loop":"iterate","action":"增强调参(SNR[0,15]+speed[0.85,1.15])+epoch=10","outcome":"effective","cer_before":0.1463,"cer_after":0.1452,"gpu_hours":0.39,"node_id":"node-3","details":"CER 14.63%→14.52%(-0.11%), 召回78.33%→80%(+1.67%), Ins 25%→22%, Del 39%→44%升高. 热词优化无效"}

{"timestamp":"2026-07-10T10:17:00Z","loop":"iterate","action":"解冻seaco_decoder训练(freeze=hotword_only)","outcome":"ineffective","cer_before":0.1452,"cer_after":0.1452,"gpu_hours":0.43,"node_id":"node-4","details":"CER/召回与node-3完全相同，解冻seaco_decoder在当前增强数据+lr下无效"}

{"timestamp":"2026-07-10T11:00:00Z","loop":"iterate","action":"lr=5e-5 + epoch=15 训练配方","outcome":"ineffective","cer_before":0.1452,"cer_after":0.1490,"gpu_hours":0.52,"node_id":"node-5","details":"CER 14.52%→14.90%(退化+0.38%), 召回80%→78%(退化-2%). 降lr导致欠拟合退化"}

{"timestamp":"2026-07-10T11:30:00Z","loop":"iterate","action":"精简增强链(关闭clipping+noise)","outcome":"effective","cer_before":0.1452,"cer_after":0.1360,"gpu_hours":0.31,"node_id":"node-6","details":"CER 14.52%→13.60%(-0.92%), 最大单轮改善! 去掉非核心噪声让模型更专注。召回微降80%→79.67%"}
