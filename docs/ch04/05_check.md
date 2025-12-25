# トークナイザー検証

## 学習目標

学習したBPEトークナイザーの**品質を検証**する方法を理解する。

## 主要概念

### 1. 学習されたトークンの確認

```python
tokenizer = BPETokenizer.load_from("storybot/merge_rules.pkl")

print("最初に学習された10個:")
for token_id in range(256, 266):
    byte_seq = tokenizer.id_to_byte[token_id]
    text = byte_seq.decode('utf-8', errors='replace')
    print(f"  ID {token_id}: '{text}'")

print("\n最後に学習された10個:")
for token_id in range(9990, 10000):
    byte_seq = tokenizer.id_to_byte[token_id]
    text = byte_seq.decode('utf-8', errors='replace')
    print(f"  ID {token_id}: '{text}'")
```

### 2. 圧縮率の測定

```python
sample_text = open("storybot/tiny_stories_train.txt").read()[:10000]

byte_count = len(sample_text.encode('utf-8'))
ids = tokenizer.encode(sample_text)
ids_count = len(ids)
compression_ratio = byte_count / ids_count

print(f"バイト数: {byte_count:,}")
print(f"トークン数: {ids_count:,}")
print(f"圧縮率: {compression_ratio:.2f}倍")
```

### 3. 異なるトークナイザーの比較

```python
# StoryBot用トークナイザー（語彙10000）
print("=== StoryBotトークナイザ ===")
tokenizer = BPETokenizer.load_from("storybot/merge_rules.pkl")
ids = tokenizer.encode(sample_text)
print(f"圧縮率: {byte_count / len(ids):.2f}倍")

# CodeBot用トークナイザー（語彙1000）
print("\n=== CodeBotトークナイザ ===")
tokenizer = BPETokenizer.load_from("codebot/merge_rules.pkl")
ids = tokenizer.encode(sample_text)
print(f"圧縮率: {byte_count / len(ids):.2f}倍")
```

## 検証のポイント

### 良いトークナイザーの指標

| 指標 | 良い状態 |
|------|---------|
| 圧縮率 | 高い（3-4倍以上） |
| 学習トークン | 意味のある単位 |
| ドメイン適合 | 対象データで高圧縮 |

### 最初に学習されるトークン例

```
ID 256: '  '    # 2スペース（インデント）
ID 257: 'th'    # 頻出の2文字
ID 258: 'the'   # 冠詞
```

### 最後に学習されるトークン例

```
ID 9998: 'beautiful'  # 長い単語
ID 9999: 'grandmother' # 複合語
```

## ドメイン特化の重要性

同じテキストでも、学習データによって圧縮率が異なる：

| トークナイザー | 学習データ | 物語テキストでの圧縮率 |
|--------------|-----------|---------------------|
| StoryBot | 物語 | 高い（ドメイン一致） |
| CodeBot | コード | 低い（ドメイン不一致） |
