# BPEチャンク処理

## 学習目標

大規模ファイルを**チャンク単位**で処理し、メモリ効率を改善する方法を理解する。

## 主要概念

### 1. チャンク境界の検出

```python
def find_chunk_boundaries(file_path, num_chunks, end_token="<|endoftext|>"):
    byte_end_token = end_token.encode("utf-8")

    with open(file_path, "rb") as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // num_chunks

        # 等間隔の境界位置
        chunk_boundaries = [i * chunk_size for i in range(num_chunks)]
        chunk_boundaries.append(file_size)

        buffer_size = 4096

        # 境界位置を終了トークンに合わせて調整
        for bi in range(1, len(chunk_boundaries) - 1):
            chunk_position = chunk_boundaries[bi]
            file.seek(chunk_position)

            while True:
                buffer = file.read(buffer_size)
                if buffer == b"":
                    chunk_boundaries[bi] = file_size
                    break

                end_position = buffer.find(byte_end_token)
                if end_position != -1:
                    chunk_boundaries[bi] = chunk_position + end_position
                    break

                chunk_position += buffer_size

    return sorted(set(chunk_boundaries))
```

### 2. チャンク単位の読み込み

```python
def train_bpe(file_path, vocab_size, end_token="<|endoftext|>"):
    chunk_boundaries = find_chunk_boundaries(file_path, num_chunks=64)

    pretoken_counts = defaultdict(int)
    with open(file_path, "rb") as f:
        for i in range(len(chunk_boundaries) - 1):
            start = chunk_boundaries[i]
            end = chunk_boundaries[i+1]

            f.seek(start)
            chunk_byte = f.read(end - start)
            chunk_text = chunk_byte.decode("utf-8", errors="ignore")

            # 特殊トークンで分割して事前トークン化
            texts = chunk_text.split(end_token)
            for text in texts:
                for pretoken in pretokenize_iter(text):
                    pretoken_counts[pretoken] += 1

    # 以降は通常のBPE学習...
```

## なぜチャンク処理が必要か

| 方法 | メモリ使用量 | 処理速度 |
|------|-------------|---------|
| 全読み込み | O(ファイルサイズ) | 速い |
| チャンク処理 | O(チャンクサイズ) | やや遅い |

大規模ファイル（数GB以上）では全読み込みが不可能。

## ポイント

1. **バイナリモード**: `"rb"` でファイルを開く
2. **終了トークン境界**: テキストの途中で切らない
3. **UTF-8デコード**: `errors="ignore"` で不正なバイトをスキップ
4. **seek/read**: 必要な部分のみ読み込み
