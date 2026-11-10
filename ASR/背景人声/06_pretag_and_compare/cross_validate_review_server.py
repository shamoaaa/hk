#!/usr/bin/env python3
"""
交叉验证人工复核工具 (Web版)
加载 cross_validate_labels.py 的输出 (validate_detail.jsonl)，
在浏览器中播放音频，展示人工/FireRed/Qwen三方标签，
提供选择按钮和可编辑的预测结果，支持剔除信号差样本。

选项:
  human    — 采用人工标签
  firered  — 采用 FireRedASR 结果
  qwen     — 采用 Qwen3-ASR 结果
  edit     — 手动编辑预测结果（可在输入框中修改）
  bad      — 信号差，剔除该样本

依赖: pip install flask

用法:
    python cross_validate_review_server.py
    python cross_validate_review_server.py --detail validate_detail.jsonl --wav-dir labeled/wav
    python cross_validate_review_server.py --category models_agree
    python cross_validate_review_server.py --port 5001
"""

import json
import argparse
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, send_file

# BASE_DIR = Path(r'E:\8.音频\03 s10\s10_data_reflow')
BASE_DIR = Path(r'E:\8.音频\03 s10\sp4_data')

app = Flask(__name__)

STATE = {
    "wav_dir": None,
    "items": [],           # validate_detail.jsonl 加载的条目列表
    "all_uids": [],        # 有序 uid 列表
    "decisions": {},       # uid -> {choice, text}
    "decision_file": None,
}

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>交叉验证复核</title>
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
    max-width: 800px; width: 95%; box-shadow: 0 8px 32px rgba(0,0,0,.4);
  }
  .progress {
    font-size: 14px; color: #888; margin-bottom: 12px; text-align: right;
  }
  .progress-bar {
    height: 4px; background: #333; border-radius: 2px; margin-bottom: 20px;
    overflow: hidden;
  }
  .progress-bar .fill {
    height: 100%; background: linear-gradient(90deg, #0f3460, #e94560);
    transition: width .3s;
  }
  .uid { font-size: 13px; color: #666; margin-bottom: 4px; word-break: break-all; }
  .meta { font-size: 12px; color: #555; margin-bottom: 16px; }
  .meta .cat { color: #e94560; font-weight: 600; }
  .meta .cer { margin-left: 12px; }

  .label-section { margin-bottom: 20px; }
  .label-row {
    display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
  }
  .label-tag {
    flex-shrink: 0; width: 80px; font-size: 12px; font-weight: 600;
    text-align: right; color: #888;
  }
  .label-text {
    flex: 1; background: #0f3460; border-radius: 8px; padding: 12px 16px;
    font-size: 18px; line-height: 1.5; min-height: 44px;
    display: flex; align-items: center; cursor: pointer;
    transition: outline .15s;
  }
  .label-text:hover { outline: 2px solid #3a86ff; }
  .label-text.selected { outline: 2px solid #06d6a0; }
  .label-text.empty { color: #666; font-style: italic; font-size: 14px; }

  audio { width: 100%; margin-bottom: 20px; outline: none; }

  .edit-section { margin-bottom: 20px; }
  .edit-section label { font-size: 13px; color: #888; display: block; margin-bottom: 6px; }
  .edit-row { display: flex; gap: 10px; }
  .edit-row input {
    flex: 1; background: #0a1628; border: 1px solid #333; border-radius: 8px;
    padding: 10px 14px; font-size: 16px; color: #eee; outline: none;
  }
  .edit-row input:focus { border-color: #3a86ff; }
  .edit-row button {
    padding: 10px 20px; border: none; border-radius: 8px;
    background: #06d6a0; color: #1a1a2e; font-weight: 600; cursor: pointer;
    font-size: 13px; white-space: nowrap;
  }
  .edit-row button:hover { background: #05c493; }

  .btn-row {
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;
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
  .btn-human   { background: #3a86ff; }
  .btn-firered { background: #e77f00; }
  .btn-qwen    { background: #c77dff; }
  .btn-edit    { background: #06d6a0; color: #1a1a2e !important; }
  .btn-bad     { background: #6b0f1a; }

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
  .current-decision {
    margin-top: 12px; font-size: 13px; color: #06d6a0; text-align: center;
    min-height: 20px;
  }
</style>
</head>
<body>
<div class="container" id="app">
  <div class="progress" id="progress"></div>
  <div class="progress-bar"><div class="fill" id="bar"></div></div>
  <div class="uid" id="uid"></div>
  <div class="meta" id="meta"></div>

  <div class="label-section">
    <div class="label-row">
      <span class="label-tag">人工标注</span>
      <div class="label-text" id="lb-human" onclick="choose('human')"></div>
    </div>
    <div class="label-row">
      <span class="label-tag">FireRedASR</span>
      <div class="label-text" id="lb-firered" onclick="choose('firered')"></div>
    </div>
    <div class="label-row">
      <span class="label-tag">Qwen3-ASR</span>
      <div class="label-text" id="lb-qwen" onclick="choose('qwen')"></div>
    </div>
  </div>

  <audio id="player" controls autoplay></audio>

  <div class="edit-section">
    <label>预测结果（可编辑）</label>
    <div class="edit-row">
      <input type="text" id="edit-input" placeholder="点击上方标签或手动输入..." />
      <button onclick="choose('edit')">确认编辑 <kbd>E</kbd></button>
    </div>
  </div>

  <div class="btn-row">
    <button class="btn-human" onclick="choose('human')">人工标注 <kbd>1</kbd></button>
    <button class="btn-firered" onclick="choose('firered')">FireRed <kbd>2</kbd></button>
    <button class="btn-qwen" onclick="choose('qwen')">Qwen <kbd>3</kbd></button>
    <button class="btn-edit" onclick="choose('edit')">编辑结果 <kbd>E</kbd></button>
    <button class="btn-bad" onclick="choose('bad')">信号差剔除 <kbd>D</kbd></button>
  </div>

  <div class="nav-row">
    <button onclick="nav('prev')">⬅ 上一条 <kbd>←</kbd></button>
    <button onclick="nav('next')">下一条 ➡ <kbd>→</kbd></button>
    <button onclick="nav('skip')">跳到未复核 <kbd>S</kbd></button>
  </div>
  <div class="current-decision" id="cur-decision"></div>
  <div class="stats" id="stats"></div>
</div>

<script>
let idx = 0, total = 0, reviewed = 0;
let currentItem = null;

async function load(i) {
  const r = await fetch('/api/item?index=' + i);
  const d = await r.json();
  if (d.done) {
    document.getElementById('lb-human').innerHTML = '<div class="done-msg">全部复核完成！</div>';
    document.getElementById('lb-firered').textContent = '';
    document.getElementById('lb-qwen').textContent = '';
    document.getElementById('player').src = '';
    document.getElementById('uid').textContent = '';
    document.getElementById('meta').textContent = '';
    document.getElementById('edit-input').value = '';
    updateStats(d);
    return;
  }
  currentItem = d;
  idx = d.index; total = d.total; reviewed = d.reviewed;

  document.getElementById('uid').textContent = d.uid;

  const catNames = {
    'models_agree': '两模型一致 人工不同',
    'all_differ': '三方都不同',
    'human_firered': '人工=FireRed Qwen不同',
    'human_qwen': '人工=Qwen FireRed不同',
    'all_agree': '三方一致'
  };
  document.getElementById('meta').innerHTML =
    `<span class="cat">${catNames[d.category] || d.category}</span>` +
    `<span class="cer">H-F: ${(d.cer_hf*100).toFixed(1)}%</span>` +
    `<span class="cer">H-Q: ${(d.cer_hq*100).toFixed(1)}%</span>` +
    `<span class="cer">F-Q: ${(d.cer_fq*100).toFixed(1)}%</span>`;

  setLabel('lb-human', d.human);
  setLabel('lb-firered', d.firered);
  setLabel('lb-qwen', d.qwen);

  // 清除选中状态
  document.querySelectorAll('.label-text').forEach(el => el.classList.remove('selected'));

  // 恢复已有决策
  if (d.decision) {
    document.getElementById('edit-input').value = d.decision.text || '';
    document.getElementById('cur-decision').textContent =
      '当前: ' + choiceLabel(d.decision.choice) + (d.decision.text ? ' → ' + d.decision.text : '');
    if (d.decision.choice === 'human') document.getElementById('lb-human').classList.add('selected');
    else if (d.decision.choice === 'firered') document.getElementById('lb-firered').classList.add('selected');
    else if (d.decision.choice === 'qwen') document.getElementById('lb-qwen').classList.add('selected');
  } else {
    document.getElementById('edit-input').value = '';
    document.getElementById('cur-decision').textContent = '';
  }

  document.getElementById('player').src = '/api/audio?uid=' + encodeURIComponent(d.uid);
  document.getElementById('progress').textContent =
    (idx + 1) + ' / ' + total + '  (已复核 ' + reviewed + ')';
  document.getElementById('bar').style.width = (reviewed / total * 100) + '%';
  updateStats(d);
}

function setLabel(id, text) {
  const el = document.getElementById(id);
  if (text) { el.textContent = text; el.classList.remove('empty'); }
  else { el.textContent = '(空)'; el.classList.add('empty'); }
}

function choiceLabel(c) {
  return {human:'人工标注', firered:'FireRedASR', qwen:'Qwen3-ASR', edit:'手动编辑', bad:'信号差剔除'}[c] || c;
}

function updateStats(d) {
  if (!d.decision_dist) return;
  const s = d.decision_dist;
  document.getElementById('stats').innerHTML =
    `<span>人工:${s.human||0}</span><span>FireRed:${s.firered||0}</span>` +
    `<span>Qwen:${s.qwen||0}</span><span>编辑:${s.edit||0}</span>` +
    `<span>剔除:${s.bad||0}</span><span>总:${d.reviewed}/${d.total}</span>`;
}

async function choose(choice) {
  if (!currentItem) return;
  let text = '';
  if (choice === 'human') text = currentItem.human || '';
  else if (choice === 'firered') text = currentItem.firered || '';
  else if (choice === 'qwen') text = currentItem.qwen || '';
  else if (choice === 'edit') text = document.getElementById('edit-input').value.trim();
  else if (choice === 'bad') text = '';

  const r = await fetch('/api/decide', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({index: idx, choice: choice, text: text})
  });
  const d = await r.json();
  load(d.next_unreviewed);
}

function nav(action) {
  if (action === 'prev') load(Math.max(0, idx - 1));
  else if (action === 'next') load(idx + 1);
  else if (action === 'skip') fetch('/api/next_unreviewed').then(r=>r.json()).then(d=>load(d.index));
}

// 点击标签行时填入编辑框
document.getElementById('lb-human').addEventListener('click', () => {
  if (currentItem) document.getElementById('edit-input').value = currentItem.human || '';
});
document.getElementById('lb-firered').addEventListener('click', () => {
  if (currentItem) document.getElementById('edit-input').value = currentItem.firered || '';
});
document.getElementById('lb-qwen').addEventListener('click', () => {
  if (currentItem) document.getElementById('edit-input').value = currentItem.qwen || '';
});

document.addEventListener('keydown', e => {
  // 编辑框聚焦时只响应 Escape
  if (document.activeElement === document.getElementById('edit-input')) {
    if (e.key === 'Escape') document.getElementById('edit-input').blur();
    return;
  }
  if (e.key === '1') choose('human');
  else if (e.key === '2') choose('firered');
  else if (e.key === '3') choose('qwen');
  else if (e.key.toLowerCase() === 'e') {
    document.getElementById('edit-input').focus();
    e.preventDefault();
  }
  else if (e.key.toLowerCase() === 'd') choose('bad');
  else if (e.key === 'ArrowLeft') nav('prev');
  else if (e.key === 'ArrowRight') nav('next');
  else if (e.key.toLowerCase() === 's') nav('skip');
});

load(-1);
</script>
</body>
</html>
"""


def load_detail(detail_path: Path, category_filter: str = None) -> list:
    """加载 validate_detail.jsonl，可按 category 过滤"""
    items = []
    with open(detail_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if category_filter and obj.get("category") != category_filter:
                    continue
                # 跳过三方一致的（无需复核）
                if not category_filter and obj.get("category") == "all_agree":
                    continue
                items.append(obj)
            except json.JSONDecodeError:
                continue
    # 按最大 CER 降序，优先复核差异大的
    items.sort(key=lambda r: max(r.get("cer_human_firered", 0),
                                  r.get("cer_human_qwen", 0)), reverse=True)
    return items


def load_decisions(decision_file: Path) -> dict:
    """加载已有复核决策"""
    decisions = {}
    if decision_file.exists():
        with open(decision_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    decisions[d["uid"]] = {"choice": d["choice"], "text": d.get("text", "")}
                except (json.JSONDecodeError, KeyError):
                    pass
    return decisions


def save_decision(decision_file: Path, uid: str, choice: str, text: str):
    """追加一条复核决策"""
    with open(decision_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps({"uid": uid, "choice": choice, "text": text},
                           ensure_ascii=False) + "\n")


def get_decision_distribution() -> dict:
    dist = {}
    for v in STATE["decisions"].values():
        c = v["choice"]
        dist[c] = dist.get(c, 0) + 1
    return dist


def find_next_unreviewed(start: int = 0) -> int:
    uids = STATE["all_uids"]
    decisions = STATE["decisions"]
    for i in range(start, len(uids)):
        if uids[i] not in decisions:
            return i
    return len(uids)


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/item')
def api_item():
    idx = int(request.args.get('index', 0))
    items = STATE["items"]
    uids = STATE["all_uids"]
    total = len(uids)

    if idx < 0:
        idx = find_next_unreviewed(0)

    if idx >= total:
        return jsonify({
            "done": True, "total": total,
            "reviewed": len(STATE["decisions"]),
            "decision_dist": get_decision_distribution(),
        })

    idx = max(0, min(idx, total - 1))
    item = items[idx]
    uid = item["uid"]

    return jsonify({
        "done": False,
        "index": idx,
        "uid": uid,
        "category": item.get("category", ""),
        "human": item.get("human", ""),
        "firered": item.get("firered", ""),
        "qwen": item.get("qwen", "").rstrip('，。！？、；：…·,.!?;:'),
        "cer_hf": item.get("cer_human_firered", 0),
        "cer_hq": item.get("cer_human_qwen", 0),
        "cer_fq": item.get("cer_firered_qwen", 0),
        "total": total,
        "reviewed": len(STATE["decisions"]),
        "decision": STATE["decisions"].get(uid),
        "decision_dist": get_decision_distribution(),
    })


@app.route('/api/audio')
def api_audio():
    uid = request.args.get('uid', '')
    wav_path = Path(STATE["wav_dir"]) / f"{uid}.wav"
    # print(f"[audio] uid={uid}  path={wav_path}  exists={wav_path.exists()}")
    if not wav_path.exists():
        return f"Not found: {wav_path}", 404
    return send_file(str(wav_path), mimetype='audio/wav')


@app.route('/api/decide', methods=['POST'])
def api_decide():
    data = request.get_json()
    idx = data.get("index", 0)
    choice = data.get("choice", "")
    text = data.get("text", "")

    # Qwen 输出常带末尾标点，选择 qwen 时自动去除
    if choice == "qwen" and text:
        text = text.rstrip('，。！？、；：…·,.!?;:')

    items = STATE["items"]
    if 0 <= idx < len(items):
        uid = items[idx]["uid"]
        STATE["decisions"][uid] = {"choice": choice, "text": text}
        save_decision(STATE["decision_file"], uid, choice, text)

    next_idx = find_next_unreviewed(idx + 1)
    return jsonify({"ok": True, "next_unreviewed": next_idx})


@app.route('/api/next_unreviewed')
def api_next_unreviewed():
    return jsonify({"index": find_next_unreviewed(0)})


def main():
    parser = argparse.ArgumentParser(description="交叉验证人工复核工具")
    parser.add_argument('--detail',
                        default=str(BASE_DIR / "cross_validate_output" / "validate_detail.jsonl"),
                        help='validate_detail.jsonl 路径')
    parser.add_argument('--wav-dir', default=str(BASE_DIR / "labeled" / "wav"),
                        help='wav 文件目录')
    parser.add_argument('--decision-file',
                        default=str(BASE_DIR / "cross_validate_output" / "review_decisions.jsonl"),
                        help='复核决策记录文件')
    parser.add_argument('--category', default=None,
                        choices=['models_agree', 'all_differ', 'human_firered', 'human_qwen'],
                        help='只复核指定类别 (默认: 除 all_agree 外全部)')
    parser.add_argument('--port', type=int, default=5001, help='端口号')
    parser.add_argument('--host', default='127.0.0.1', help='监听地址')
    args = parser.parse_args()

    detail_path = Path(args.detail)
    if not detail_path.exists():
        print(f"详情文件不存在: {detail_path}")
        print("请先运行 cross_validate_labels.py 生成交叉验证结果")
        return

    wav_dir = Path(args.wav_dir)
    if not wav_dir.exists():
        print(f"音频目录不存在: {wav_dir}")
        return

    STATE["wav_dir"] = str(wav_dir)
    STATE["items"] = load_detail(detail_path, args.category)
    STATE["all_uids"] = [item["uid"] for item in STATE["items"]]
    STATE["decision_file"] = Path(args.decision_file)
    STATE["decisions"] = load_decisions(STATE["decision_file"])

    cat_desc = args.category or "全部(除 all_agree)"
    print(f"详情文件: {detail_path}")
    print(f"音频目录: {wav_dir}")
    print(f"筛选类别: {cat_desc}")
    print(f"待复核数: {len(STATE['items'])}")
    print(f"已复核:   {len(STATE['decisions'])}")
    print(f"决策文件: {STATE['decision_file']}")
    print(f"\n打开浏览器访问: http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
