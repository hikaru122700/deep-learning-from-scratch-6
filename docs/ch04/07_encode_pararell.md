# エンコード並列処理

## 学習目標

大規模ファイルのエンコードを**マルチプロセス**で高速化する方法を理解する。

## 主要概念

### 1. 処理フロー

```
大規模テキストファイル
    ↓ チャンク分割
[chunk1, chunk2, chunk3, ...]
    ↓ 並列エンコード（各プロセス）
[cache1.npy, cache2.npy, ...]
    ↓ 結合
output.bin
    ↓ キャッシュ削除
完了
```

### 2. チャンクエンコード関数

```python
def _encode_chunk(self, args):
    file_path, start, end, cache_dir, chunk_idx = args

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_byte = f.read(end - start)
        chunk_text = chunk_byte.decode("utf-8", errors="ignore")

        ids = self.encode(chunk_text)

    # キャッシュファイルに保存
    cache_file = os.path.join(cache_dir, f"chunk_{chunk_idx:05d}.npy")
    np.array(ids, dtype=np.uint16).tofile(cache_file)

    return cache_file, len(ids)
```

### 3. ファイルエンコード

```python
def encode_file(self, file_path, output_file,
                num_processes=4, num_chunks=64, cache_dir="bpe_cache"):

    os.makedirs(cache_dir, exist_ok=True)

    try:
        # チャンク情報の準備
        chunk_boundaries = find_chunk_boundaries(file_path, num_chunks)
        chunk_info_list = [
            (file_path, chunk_boundaries[i], chunk_boundaries[i+1], cache_dir, i)
            for i in range(len(chunk_boundaries) - 1)
        ]

        # 並列エンコード
        with Pool(processes=num_processes) as pool:
            cache_results = list(tqdm(
                pool.imap(self._encode_chunk, chunk_info_list),
                desc="Encoding chunks"
            ))

        # 結果の統合
        cache_files = [r[0] for r in cache_results]
        token_counts = [r[1] for r in cache_results]
        total_tokens = sum(token_counts)

        # memmapファイルを作成
        arr = np.memmap(output_file, dtype=np.uint16, mode='w+', shape=(total_tokens,))

        idx = 0
        for cache_file in cache_files:
            chunk_data = np.fromfile(cache_file, dtype=np.uint16)
            arr[idx : idx + len(chunk_data)] = chunk_data
            idx += len(chunk_data)

        arr.flush()

    finally:
        shutil.rmtree(cache_dir)  # キャッシュ削除

    return total_tokens
```

### 4. 使用例

```python
if __name__ == '__main__':
    tokenizer = BPETokenizer.load_from("storybot/merge_rules.pkl")

    tokenizer.encode_file(
        "storybot/tiny_stories_train.txt",
        "storybot/tiny_stories_train.bin",
        num_processes=8
    )

    tokenizer.encode_file(
        "storybot/tiny_stories_valid.txt",
        "storybot/tiny_stories_valid.bin",
        num_processes=8
    )
```

## メモリ効率

| 方式 | メモリ使用量 |
|------|-------------|
| 全メモリ | O(ファイルサイズ + 全トークン) |
| キャッシュ方式 | O(チャンクサイズ) |

## ポイント

1. **キャッシュファイル**: 各チャンクを個別に保存
2. **memmap**: 大規模配列をディスク上で扱う
3. **finally節**: エラー時もキャッシュを確実に削除
4. **uint16**: 語彙サイズ65535以下なら十分
