#!/usr/bin/env python3
"""
音频人工审核打分工具 (Web版)
本地启动 Flask 服务，浏览器中播放音频、显示标签、按钮打分。

打分标准:
  -1 - 信号很差
   0 - 标签错误
   1 - 语音含有萤石字段
   2 - 有多人说话
   3 - 信号一般
   4 - 信号很好

依赖: pip install flask

用法:
    python audio_review_server.py --wav-dir labeled/wav --label-file labeled/text.txt
    python audio_review_server.py --wav-dir unlabeled
    python audio_review_server.py --port 8080
"""

import os
import json
import argparse
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, send_file

BASE_DIR = Path(r'E:\8.音频\03 s10\s10_data_reflow')

app = Flask(__name__)

# 全局状态，启动时初始化
STATE = {
    "wav_dir": None,
    "labels": {},          # uid -> text
    "all_uids": [],        # 有序uid列表
    "scores": {},          # uid -> score (从文件加载)
    "score_file": None,
    "current_index": 0,    # 当前未打分的起始位置
}

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>音频审核打分</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, "Microsoft YaHei", sans-serif;
    background: #1a1a2e; color: #eee;
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh;
  }
  .container {
    background: #16213e; border-radius: 16px; padding: 40px;
    max-width: 720px; width: 95%; box-shadow: 0 8px 32px rgba(0,0,0,.4);
  }
  .progress {
    font-size: 14px; color: #888; margin-bottom: 12px; text-align: right;
  }
  .progress-bar {
    height: 4px; background: #333; border-radius: 2px; margin-bottom: 24px;
    overflow: hidden;
  }
  .progress-bar .fill {
    height: 100%; background: linear-gradient(90deg, #0f3460, #e94560);
    transition: width .3s;
  }
  .uid { font-size: 13px; color: #666; margin-bottom: 8px; word-break: break-all; }
  .label-box {
    background: #0f3460; border-radius: 10px; padding: 20px;
    font-size: 22px; text-align: center; margin-bottom: 20px;
    min-height: 60px; display: flex; align-items: center; justify-content: center;
    line-height: 1.5;
  }
  .label-box.no-label { color: #888; font-style: italic; font-size: 16px; }
  audio { width: 100%; margin-bottom: 24px; outline: none; }
  .btn-row {
    display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px;
    margin-bottom: 16px;
  }
  .btn-row button {
    padding: 14px 0; border: none; border-radius: 10px;
    font-size: 13px; font-weight: 600; cursor: pointer;
    transition: transform .1s, box-shadow .2s;
    color: #fff;
  }
  .btn-row button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,.3); }
  .btn-row button:active { transform: translateY(0); }
  .btn-n1 { background: #6b0f1a; }
  .btn-0 { background: #e94560; }
  .btn-1 { background: #e77f00; }
  .btn-2 { background: #c77dff; }
  .btn-3 { background: #3a86ff; }
  .btn-4 { background: #06d6a0; }
  .nav-row { display: flex; gap: 10px; }
  .nav-row button {
    flex: 1; padding: 10px; border: 1px solid #444; border-radius: 8px;
    background: transparent; color: #aaa; cursor: pointer; font-size: 13px;
  }
  .nav-row button:hover { background: #1a1a2e; color: #fff; }
  .stats {
    margin-top: 20px; font-size: 13px; color: #666;
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 4px;
  }
  .done-msg {
    text-align: center; padding: 40px; font-size: 18px; color: #06d6a0;
  }
  kbd {
    display: inline-block; background: #333; border-radius: 4px;
    padding: 1px 6px; font-size: 11px; color: #aaa; margin-left: 2px;
  }
</style>
</head>
<body>
<div class="container" id="app">
  <div class="progress" id="progress"></div>
  <div class="progress-bar"><div class="fill" id="bar"></div></div>
  <div class="uid" id="uid"></div>
  <div class="label-box" id="label"></div>
  <audio id="player" controls autoplay></audio>
  <div class="btn-row">
    <button class="btn-n1" onclick="score(-1)">-1 信号很差<kbd>Q</kbd></button>
    <button class="btn-0" onclick="score(0)">0 标签错误<kbd>0</kbd></button>
    <button class="btn-1" onclick="score(1)">1 萤石字段<kbd>1</kbd></button>
    <button class="btn-2" onclick="score(2)">2 多人说话<kbd>2</kbd></button>
    <button class="btn-3" onclick="score(3)">3 信号一般<kbd>3</kbd></button>
    <button class="btn-4" onclick="score(4)">4 信号很好<kbd>4</kbd></button>
  </div>
  <div class="nav-row">
    <button onclick="nav('prev')">⬅ 上一条 <kbd>←</kbd></button>
    <button onclick="nav('next')">下一条 ➡ <kbd>→</kbd></button>
    <button onclick="nav('skip')">跳到未打分 <kbd>S</kbd></button>
  </div>
  <div class="stats" id="stats"></div>
</div>

<script>
let idx = 0, total = 0, scored = 0;

async function load(i) {
  const r = await fetch('/api/item?index=' + i);
  const d = await r.json();
  if (d.done) {
    document.getElementById('label').innerHTML = '<div class="done-msg">全部审核完成！</div>';
    document.getElementById('player').src = '';
    document.getElementById('uid').textContent = '';
    updateStats(d);
    return;
  }
  idx = d.index; total = d.total; scored = d.scored;
  document.getElementById('uid').textContent = d.uid;
  const lb = document.getElementById('label');
  if (d.label) { lb.textContent = d.label; lb.className = 'label-box'; }
  else { lb.textContent = '(无标签)'; lb.className = 'label-box no-label'; }
  document.getElementById('player').src = '/api/audio?uid=' + encodeURIComponent(d.uid);
  document.getElementById('progress').textContent =
    (idx + 1) + ' / ' + total + '  (已打分 ' + scored + ')';
  document.getElementById('bar').style.width = (scored / total * 100) + '%';
  updateStats(d);
}

function updateStats(d) {
  if (!d.score_dist) return;
  const s = d.score_dist;
  document.getElementById('stats').innerHTML =
    `<span>-1差:${s['-1']||0}</span><span>0-标签错:${s['0']||0}</span>` +
    `<span>1-萤石:${s['1']||0}</span><span>2-多人:${s['2']||0}</span>` +
    `<span>3-一般:${s['3']||0}</span><span>4-很好:${s['4']||0}</span>` +
    `<span>总:${d.scored}/${d.total}</span>`;
}

async function score(v) {
  const r = await fetch('/api/score', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({index: idx, score: v})
  });
  const d = await r.json();
  // 自动跳到下一条未打分的
  load(d.next_unscored);
}

function nav(action) {
  if (action === 'prev') load(Math.max(0, idx - 1));
  else if (action === 'next') load(idx + 1);
  else if (action === 'skip') fetch('/api/next_unscored').then(r=>r.json()).then(d=>load(d.index));
}

document.addEventListener('keydown', e => {
  if (e.key >= '0' && e.key <= '4') score(parseInt(e.key));
  else if (e.key.toLowerCase() === 'q') score(-1);
  else if (e.key === 'ArrowLeft') nav('prev');
  else if (e.key === 'ArrowRight') nav('next');
  else if (e.key.toLowerCase() === 's') nav('skip');
});

load(-1);  // -1 表示从第一个未打分的开始
</script>
</body>
</html>
"""


def load_labels(label_file: str) -> dict:
    """加载标签文件，返回 {uid: text}"""
    labels = {}
    if not label_file or not Path(label_file).exists():
        return labels
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) >= 2:
                labels[parts[0]] = parts[1]
    return labels


def load_scores(score_file: Path) -> dict:
    """加载已有打分"""
    scores = {}
    if score_file.exists():
        with open(score_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    scores[d["uid"]] = d["score"]
                except (json.JSONDecodeError, KeyError):
                    pass
    return scores


def save_score(score_file: Path, uid: str, score_val: int):
    """追加一条打分记录"""
    with open(score_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps({"uid": uid, "score": score_val}, ensure_ascii=False) + "\n")


def get_score_distribution() -> dict:
    dist = {}
    for v in STATE["scores"].values():
        dist[str(v)] = dist.get(str(v), 0) + 1
    return dist


def find_next_unscored(start: int = 0) -> int:
    """从 start 开始找下一个未打分的索引"""
    uids = STATE["all_uids"]
    scores = STATE["scores"]
    for i in range(start, len(uids)):
        if uids[i] not in scores:
            return i
    # 全部打完了，返回最后一个
    return len(uids)


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/item')
def api_item():
    idx = int(request.args.get('index', 0))
    uids = STATE["all_uids"]
    total = len(uids)

    # -1 表示自动跳到第一个未打分的
    if idx < 0:
        idx = find_next_unscored(0)

    if idx >= total:
        return jsonify({
            "done": True, "total": total,
            "scored": len(STATE["scores"]),
            "score_dist": get_score_distribution(),
        })

    idx = max(0, min(idx, total - 1))
    uid = uids[idx]
    label = STATE["labels"].get(uid, "")

    return jsonify({
        "done": False,
        "index": idx,
        "uid": uid,
        "label": label,
        "total": total,
        "scored": len(STATE["scores"]),
        "current_score": STATE["scores"].get(uid),
        "score_dist": get_score_distribution(),
    })


@app.route('/api/audio')
def api_audio():
    uid = request.args.get('uid', '')
    wav_path = Path(STATE["wav_dir"]) / f"{uid}.wav"
    if not wav_path.exists():
        return "Not found", 404
    return send_file(str(wav_path), mimetype='audio/wav')


@app.route('/api/score', methods=['POST'])
def api_score():
    data = request.get_json()
    idx = data.get("index", 0)
    score_val = data.get("score", -1)

    uids = STATE["all_uids"]
    if 0 <= idx < len(uids):
        uid = uids[idx]
        STATE["scores"][uid] = score_val
        save_score(STATE["score_file"], uid, score_val)

    next_idx = find_next_unscored(idx + 1)
    return jsonify({"ok": True, "next_unscored": next_idx})


@app.route('/api/next_unscored')
def api_next_unscored():
    return jsonify({"index": find_next_unscored(0)})


def main():
    parser = argparse.ArgumentParser(description="音频人工审核打分工具")
    parser.add_argument('--wav-dir', default=str(BASE_DIR / "labeled" / "wav"),
                        help='wav文件目录')
    parser.add_argument('--label-file', default=str(BASE_DIR / "labeled" / "text.txt"),
                        help='标签文件路径 (可选)')
    parser.add_argument('--score-file', default=str(BASE_DIR / "review_scores.jsonl"),
                        help='打分记录文件')
    parser.add_argument('--port', type=int, default=5000, help='端口号')
    parser.add_argument('--host', default='127.0.0.1', help='监听地址')
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    if not wav_dir.exists():
        print(f"音频目录不存在: {wav_dir}")
        return

    # 初始化
    STATE["wav_dir"] = str(wav_dir)
    STATE["labels"] = load_labels(args.label_file)
    STATE["score_file"] = Path(args.score_file)
    STATE["scores"] = load_scores(STATE["score_file"])

    # 扫描wav
    all_wavs = sorted(wav_dir.glob("*.wav"))
    STATE["all_uids"] = [w.stem for w in all_wavs]

    print(f"音频目录: {wav_dir}")
    print(f"音频数量: {len(STATE['all_uids'])}")
    print(f"标签数量: {len(STATE['labels'])}")
    print(f"已打分:   {len(STATE['scores'])}")
    print(f"打分文件: {STATE['score_file']}")
    print(f"\n打开浏览器访问: http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
