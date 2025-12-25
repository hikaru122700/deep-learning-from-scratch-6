# BPE並列処理

## 学習目標

**マルチプロセス**でBPE学習を高速化する方法を理解する。

## 主要概念

### 1. 並列化の戦略

事前トークン化フェーズを並列化：
- 各プロセスがチャンクを担当
- 結果を集約

### 2. チャンク処理関数

```python
def pretoken_chunk(args):
    file_path, start, end, end_token = args
    pretoken_counts = defaultdict(int)

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_byte = f.read(end - start)
        chunk_text = chunk_byte.decode("utf-8", errors="ignore")

        texts = chunk_text.split(end_token)
        for text in texts:
            for pretoken in pretokenize_iter(text):
                pretoken_counts[pretoken] += 1

    return pretoken_counts
```

### 3. 並列BPE学習

```python
def train_bpe(file_path, vocab_size, end_token="<|endoftext|>",
              num_processes=8, num_chunks=64):

    # チャンク情報の準備
    chunk_boundaries = find_chunk_boundaries(file_path, num_chunks)
    chunk_info_list = [
        (file_path, chunk_boundaries[i], chunk_boundaries[i+1], end_token)
        for i in range(len(chunk_boundaries) - 1)
    ]

    # 並列処理
    with Pool(processes=num_processes) as pool:
        all_results = list(tqdm(
            pool.imap(pretoken_chunk, chunk_info_list),
            total=len(chunk_info_list),
            desc="Pretokenizing"
        ))

    # 結果を統合
    pretoken_counts = defaultdict(int)
    for chunk_result in all_results:
        for pretoken, count in chunk_result.items():
            pretoken_counts[pretoken] += count

    # 以降は通常のBPE学習（シングルプロセス）...
```

### 4. 使用例

```python
if __name__ == '__main__':  # Windows対応
    vocab_size = 10000
    file_path = "storybot/tiny_stories_train.txt"
    merge_rules = train_bpe(file_path, vocab_size, num_processes=8)

    with open("storybot/merge_rules.pkl", "wb") as f:
        pickle.dump(merge_rules, f)
```

## 並列化のポイント

| フェーズ | 並列化 | 理由 |
|----------|--------|------|
| 事前トークン化 | ○ | チャンク間で独立 |
| マージルール学習 | × | 逐次的な依存関係 |

## 速度向上

```
8プロセスの場合: 約 4-6倍の高速化
（オーバーヘッドがあるため、完全に8倍にはならない）
```

## ポイント

1. **`if __name__ == '__main__'`**: Windowsでのマルチプロセス対応
2. **Pool.imap**: メモリ効率の良いマッピング
3. **tqdm統合**: 並列処理の進捗表示
4. **結果の統合**: 各プロセスの結果を合算
