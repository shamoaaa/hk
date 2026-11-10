#!/usr/bin/env python3
"""
从标签文件生成热词表。

提取方式:
  1. jieba 分词统计词频
  2. 过滤停用词和单字
  3. 提取高频 n-gram (bigram/trigram)
  4. 提取人名、地名等命名实体模式
  5. 按频次排序输出

依赖: pip install jieba

用法:
    python gen_hotwords.py
    python gen_hotwords.py --top 500
    python gen_hotwords.py --min-freq 5
    python gen_hotwords.py --input labeled/text.txt
"""

import re
import json
import argparse
from pathlib import Path
from collections import Counter

import jieba

BASE_DIR = Path(r'E:\8.音频\03 s10\s10_data_reflow')
DEFAULT_INPUT = BASE_DIR / "labeled" / "text.txt"
OUTPUT_DIR = BASE_DIR / "hotwords"

# 停用词（常见虚词、语气词、单字助词）
STOPWORDS = set(
    "的 了 是 在 我 你 他 她 它 们 这 那 有 不 也 就 都 和 与 或 但 而 "
    "吗 呢 吧 啊 哦 嗯 哈 呀 哎 喂 嘿 噢 唉 哇 嗷 额 "
    "把 被 让 给 对 从 到 向 往 跟 比 为 因 所以 如果 虽然 "
    "个 些 么 什么 怎么 怎样 哪 哪里 哪个 谁 多少 几 "
    "会 能 可以 要 想 得 着 过 地 很 太 最 更 还 又 再 "
    "上 下 里 中 前 后 左 右 大 小 好 去 来 说 看 做 "
    "一 二 三 四 五 六 七 八 九 十 百 千 万 亿 "
    "第 年 月 日 号 点 分 秒 时 天 周 "
    "请 帮 让我 可不可以 能不能".split()
)


def load_texts(input_file: Path) -> list:
    """加载标签文件，返回纯文本列表"""
    texts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) >= 2:
                texts.append(parts[1])
    return texts


def extract_words(texts: list, min_freq: int) -> Counter:
    """jieba 分词 + 词频统计"""
    word_freq = Counter()
    for text in texts:
        words = jieba.lcut(text)
        for w in words:
            w = w.strip()
            if len(w) < 2:
                continue
            if w in STOPWORDS:
                continue
            # 过滤纯数字、纯标点
            if re.match(r'^[\d\s\W]+$', w):
                continue
            word_freq[w] += 1
    # 过滤低频
    return Counter({w: c for w, c in word_freq.items() if c >= min_freq})


def extract_ngrams(texts: list, n: int, min_freq: int) -> Counter:
    """提取字符级 n-gram 热词"""
    ngram_freq = Counter()
    for text in texts:
        # 去除标点空格
        clean = re.sub(r'[\s\W]+', '', text)
        for i in range(len(clean) - n + 1):
            gram = clean[i:i + n]
            ngram_freq[gram] += 1
    return Counter({g: c for g, c in ngram_freq.items() if c >= min_freq})


def extract_full_commands(texts: list, min_freq: int) -> Counter:
    """提取高频完整指令（整句去重统计）"""
    return Counter({t: c for t, c in Counter(texts).items() if c >= min_freq})


def extract_name_patterns(texts: list) -> list:
    """提取 '打电话给X' / '拨打X' 等模式中的人名"""
    name_freq = Counter()
    patterns = [
        r'(?:打电话给|拨打电话给|拨打|呼叫|联系|发消息给|发短信给)(.{1,6}?)(?:$|的|电话)',
        r'(?:给)(.{1,6}?)(?:打电话|发消息|发短信)',
    ]
    for text in texts:
        for pat in patterns:
            matches = re.findall(pat, text)
            for m in matches:
                m = m.strip()
                if len(m) >= 2 and not re.match(r'^[\d\s]+$', m):
                    name_freq[m] += 1
    return name_freq


def main():
    parser = argparse.ArgumentParser(description="生成热词表")
    parser.add_argument('--input', default=str(DEFAULT_INPUT), help='标签文件路径')
    parser.add_argument('--top', type=int, default=300, help='输出前N个热词')
    parser.add_argument('--min-freq', type=int, default=3, help='最低词频')
    args = parser.parse_args()

    input_file = Path(args.input)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"输入文件: {input_file}")
    texts = load_texts(input_file)
    print(f"加载文本: {len(texts)} 条")

    # 1. 分词热词
    print("分词统计中...")
    word_freq = extract_words(texts, args.min_freq)
    print(f"  词汇数: {len(word_freq)} (freq >= {args.min_freq})")

    # 2. bigram / trigram
    print("提取 n-gram...")
    bigrams = extract_ngrams(texts, 2, args.min_freq * 2)
    trigrams = extract_ngrams(texts, 3, args.min_freq * 2)
    fourgrams = extract_ngrams(texts, 4, args.min_freq)

    # 3. 高频完整指令
    print("统计高频指令...")
    commands = extract_full_commands(texts, args.min_freq)
    print(f"  高频指令: {len(commands)} 条")

    # 4. 人名提取
    print("提取人名...")
    names = extract_name_patterns(texts)
    print(f"  人名: {len(names)} 个")

    # --- 输出热词表 (纯文本，每行一个词，FunASR hotword 格式) ---
    hotword_path = OUTPUT_DIR / "hotwords.txt"
    # 合并所有来源，按频次排序去重
    merged = Counter()
    merged.update(word_freq)
    # n-gram 中过滤掉已有的分词结果，避免重复
    for gram_counter in [bigrams, trigrams, fourgrams]:
        for g, c in gram_counter.items():
            if g not in merged:
                merged[g] += c

    top_words = merged.most_common(args.top)
    with open(hotword_path, 'w', encoding='utf-8') as f:
        for word, freq in top_words:
            f.write(f"{word}\n")
    print(f"\n热词表 (top {args.top}): {hotword_path}")

    # --- 带频次的详细版本 ---
    detail_path = OUTPUT_DIR / "hotwords_detail.txt"
    with open(detail_path, 'w', encoding='utf-8') as f:
        f.write(f"# 热词表 (共 {len(top_words)} 词, 来源: {input_file.name})\n")
        f.write(f"# 格式: 热词 \\t 频次\n\n")
        for word, freq in top_words:
            f.write(f"{word}\t{freq}\n")
    print(f"热词详情: {detail_path}")

    # --- 高频指令 ---
    cmd_path = OUTPUT_DIR / "frequent_commands.txt"
    top_cmds = commands.most_common(200)
    with open(cmd_path, 'w', encoding='utf-8') as f:
        f.write(f"# 高频完整指令 (top 200)\n# 格式: 指令 \\t 频次\n\n")
        for cmd, freq in top_cmds:
            f.write(f"{cmd}\t{freq}\n")
    print(f"高频指令: {cmd_path}")

    # --- 人名表 ---
    name_path = OUTPUT_DIR / "contact_names.txt"
    top_names = names.most_common(200)
    with open(name_path, 'w', encoding='utf-8') as f:
        for name, freq in top_names:
            f.write(f"{name}\t{freq}\n")
    print(f"联系人名: {name_path}")

    # --- JSON 汇总 ---
    summary = {
        "total_texts": len(texts),
        "unique_words": len(word_freq),
        "hotword_count": len(top_words),
        "top20_words": [{"word": w, "freq": c} for w, c in top_words[:20]],
        "top20_commands": [{"cmd": c, "freq": f} for c, f in top_cmds[:20]],
        "top20_names": [{"name": n, "freq": f} for n, f in top_names[:20]],
    }
    summary_path = OUTPUT_DIR / "hotwords_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"汇总JSON: {summary_path}")

    # 打印 top 30 预览
    print(f"\n--- Top 30 热词 ---")
    for i, (w, c) in enumerate(top_words[:30], 1):
        print(f"  {i:>3}. {w:<12s} {c}")


if __name__ == "__main__":
    main()
