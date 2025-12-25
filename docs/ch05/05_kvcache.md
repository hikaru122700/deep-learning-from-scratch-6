# KVキャッシュ（KV Cache）

## 学習目標

推論時の高速化手法である**KVキャッシュ**を理解する。

## 主要概念

### 1. 問題点

自己回帰生成では、新しいトークンを生成するたびに：
- 全トークンでQ, K, Vを再計算
- 以前のK, Vは変わらないのに無駄

### 2. KVキャッシュの仕組み

以前に計算したK, Vをキャッシュし、新しいトークンの分だけ計算：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, ...):
        # ...
        self.k_cache = None
        self.v_cache = None
        self.cache_offset = 0

    def forward(self, x, use_cache=False):
        Q = self.W_q(x).view(B, C, H, D).transpose(1, 2)
        K = self.W_k(x).view(B, C, H, D).transpose(1, 2)
        V = self.W_v(x).view(B, C, H, D).transpose(1, 2)

        # RoPE（offsetを考慮）
        if self.rope is not None:
            if use_cache:
                Q = self.rope(Q, self.cache_offset)
                K = self.rope(K, self.cache_offset)
            else:
                Q = self.rope(Q)
                K = self.rope(K)

        # KVキャッシュの処理
        if use_cache:
            if self.k_cache is None:
                self.k_cache = K
                self.v_cache = V
            else:
                self.k_cache = torch.cat([self.k_cache, K], dim=2)
                self.v_cache = torch.cat([self.v_cache, V], dim=2)

            self.cache_offset += C
            K = self.k_cache
            V = self.v_cache

        # 通常のAttention計算（マスクはキャッシュ使用時は不要）
        # ...
```

### 3. RoPEのoffset対応

```python
def forward(self, x, offset=0):
    cos = self.cos_cache[offset:offset + context_len]
    sin = self.sin_cache[offset:offset + context_len]
    # ...
```

新しいトークンの位置は offset から始まる。

### 4. 生成関数の比較

```python
# キャッシュなし：毎回全トークンを処理
def generate_without_cache(model, start_ids, max_new_tokens):
    ids = start_ids
    for _ in range(max_new_tokens):
        logits = model(ids, use_cache=False)  # 全トークン
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids

# キャッシュあり：新しいトークンのみ処理
def generate_with_cache(model, start_ids, max_new_tokens):
    ids = start_ids
    next_id = start_ids
    for _ in range(max_new_tokens):
        logits = model(next_id, use_cache=True)  # 1トークン
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids
```

### 5. 速度比較

```python
time_without = measure_generation_time(model, start_ids, use_cache=False)
time_with = measure_generation_time(model, start_ids, use_cache=True)

print(f"KV-Cacheなし: {time_without:.2f}秒")
print(f"KV-Cacheあり: {time_with:.2f}秒")
print(f"高速化率: {time_without / time_with:.1f}倍")
```

## 計算量の比較

| 方式 | 1トークン生成の計算量 |
|------|---------------------|
| キャッシュなし | O(n²)（全トークン） |
| キャッシュあり | O(n)（新トークンのみ） |

n トークン生成時：
- キャッシュなし: O(n³)
- キャッシュあり: O(n²)

## ポイント

1. **clear_cache()**: 新しい生成開始時にキャッシュをクリア
2. **マスク不要**: キャッシュ使用時は過去のトークンのみ参照
3. **メモリ増加**: キャッシュ分のメモリが必要
