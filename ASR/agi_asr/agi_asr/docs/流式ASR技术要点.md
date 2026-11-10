# 流式 ASR 技术要点

## 1. 通信协议架构

本项目的流式 ASR 采用**三层协议混合**架构：

| 链路 | 协议 | 特点 |
|------|------|------|
| 客户端 → ms_manager | REST HTTP/JSON | 多次 POST 模拟流式，非长连接 |
| ms_manager → backend | gRPC | 真正的流式通信 |
| ms_manager → 回调服务 | REST HTTP/JSON | 异步 POST 推送结果 |

对外暴露的是 REST + HTTP/JSON 接口，通过 `request_id` 关联多次 HTTP 请求维持"流式会话"状态。

**优点：** 兼容性强，任何 HTTP 客户端都能调用。
**缺点：** 每帧都有 HTTP 建连开销，延迟比 WebSocket/gRPC 高。

---

## 2. 丢包与乱序处理

**结论：项目未做应用层排序/重传处理。**

原因：
- TCP 协议天然保证传输层有序可靠
- ms_manager 框架对同一 request_id 的请求做了串行化
- `sequence_index` 仅用于首帧判断和调试标记，非排序用途
- 部署在内网/局域网，网络环境可控

如果换成 UDP 或 WebSocket（可能被中间件打乱），才需要应用层排序。

---

## 3. 2-Pass 策略详解

### 3.1 核心思想

| 阶段 | 模型 | 解码策略 | 延迟 | 精度 |
|------|------|---------|------|------|
| Pass-1 (流式) | ParaformerOnline | 贪心 argmax | 低（每 480ms） | 较低 |
| Pass-2 (纠正) | SeacoParaformer | BeamSearch + 热词 | 高（等全部音频） | 高 |

### 3.2 stride_size 计算

```
stride_size = 60ms × chunk_size[1] × 16000Hz × 2bytes
            = 60 × 8 / 1000 × 16000 × 2
            = 15360 bytes = 480ms 音频
```

### 3.3 decode_beg / decode_end 双指针机制

```
data_online: [====已处理====|====待处理====|====新到数据====]
              0          decode_beg      decode_end

待处理长度 = decode_end - decode_beg
当 ≥ stride_size 时触发流式推理
Pass-2 纠正后归零（但 data_online 不清空，供 SV 使用）
```

### 3.4 为什么 Pass-2 后还要调 inference_online(is_final=True)

流式模型内部维护跨 chunk 的 cache（encoder/CIF/decoder FSMN），这些 cache 不会因为外部变量 `is_final=True` 就自动清空。必须显式调用 `inference_online` 并传入 `is_final=True` 触发模型内部的 cache 重置。

---

## 4. 流式结束的两种触发方式

### 4.1 客户端主动结束

```json
{"sequence_end": true, "data_info": [...]}
```

客户端在最后一帧设置 `sequence_end=true`，服务端读取后赋值给 `is_final`。

### 4.2 服务端 VAD 主动结束

VAD 检测到语音终止点（静音超过 max_end_sil）后：
1. 截取有效语音段
2. 强制 `is_final = True`
3. 触发 Pass-2 离线纠正
4. 响应中带 `sequence_end=True`

客户端收到后停止发帧。如果因网络延迟多发了几帧，ms_manager 通过 early_stop 机制释放 channel 并丢弃后续请求。

---

## 5. 2-Pass 边界情况：流式有结果但最终为空

| 场景 | 原因 | 客户端感知 |
|------|------|-----------|
| 声纹验证触发静默 | 非同一人 + 响度差 ≥ 6.5dB | 先收到中间文本，最后收到空结果 |
| KWS 唤醒词验证失败 | 云端 KWS 判定非真正唤醒词 | 会话被强制终止 |
| 离线模型判定为噪声 | SeacoParaformer 综合判断无效语音 | 最终结果为空 |
| VAD 截断后音频过短 | 有效段 < 3200 bytes (100ms) | AudioDecodeFail 错误 |

**客户端处理原则：** 以 `stage_flag=1` 的结果为准。中间结果（stage_flag=0）仅供即时反馈显示，最终结果为空时应丢弃之前的中间结果。

---

## 6. 辅助模块协作

### 6.1 VAD + 语义完整性联动

```
FSMN-VAD（声学层面）检测静音 → max_end_sil 控制等待时长
BERT 语义完整性（语义层面）→ 动态调整 max_end_sil
  - "打开空调"（完整）→ ~500ms 就结束
  - "帮我把"（不完整）→ ~1300ms 才结束
```

### 6.2 KWS 唤醒词二次验证

触发条件（三个同时满足）：
- `audio_wakeup_type == 0`（主唤醒词唤醒）
- `business_type in wakeup_refilter_type`（业务类型启用）
- `wakeup_audio_length >= 48000`（唤醒音频 ≥ 1.5s）

"你好萤石"正常发音约 0.8~1.2s，加上前后静音缓冲约 1.5s。

### 6.3 声纹验证（SV）三关判断

```
第一关：音频长度 > 0.5s（太短无法提取可靠声纹）
第二关：ERes2NetV2 余弦相似度 ≤ 0.3（非同一人）
第三关：LUFS 响度差 ≥ 6.5dB（唤醒音明显比命令音响）
→ 三关全过才触发静默
```

为什么需要响度差：区分"家人正常使用"（声纹不同但响度差小，放行）和"电视背景声误触发"（声纹不同且唤醒音比电视声大很多，静默）。

---

## 7. Opus 音频编解码

| 维度 | PCM (raw) | Opus | 压缩比 |
|------|-----------|------|--------|
| 16kHz 1秒音频 | 32 KB | ~4 KB | ~8× |
| 网络带宽 | 256 kbps | ~32 kbps | 节省 87% |

解码方式：固定 CBR 码率，每 80 字节 Opus 帧 → 解码为 640 字节 PCM（20ms 音频，320 samples × int16）。

---

## 8. 回调地址配置

回调地址在 `Cloud_TP.json` 中配置：
```json
{
    "interval": 6000,
    "stream_result_url": "http://10.86.33.252:25040/stream_result"
}
```

读取和使用逻辑在 ms_manager 框架包中（`/opt/conda/lib/python3.9/site-packages/ms_manager/utils/upload.py`），非业务代码。

---

## 9. 常见错误码

| 错误码 | 含义 | 常见原因 |
|--------|------|---------|
| 7042 | 回调超时 | stream_result_url 目标服务未响应 |
| 7080 | 回调上报失败 | POST stream_result 连续重试失败 |
| 60205 | 服务内部错误 | 回调目标服务自身报错 |

这些错误码属于 ms_manager 框架内部定义，非业务代码。
