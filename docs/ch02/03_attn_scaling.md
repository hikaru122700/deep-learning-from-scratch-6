# Attentionスケーリング（Attention Scaling）

## 学習目標

高次元でAttentionが不安定になる問題と、**スケーリング**による解決策を理解する。

## 主要概念

### 1. 問題：高次元での内積

次元数が大きいと、内積の値も大きくなる：

```python
d = 10
q = np.random.randn(d)
k = np.random.randn(d)

dot_product = np.dot(q, k)  # 大きな値になりやすい
```

### 2. 統計的な検証

```python
d = 10
num_samples = 10000

dot_products = []
for _ in range(num_samples):
    q = np.random.randn(d)
    k = np.random.randn(d)
    dot_products.append(np.dot(q, k))

print("分散:", np.var(dot_products))  # ≈ d（次元数に比例）
```

### 3. ソフトマックスの問題

```python
x = torch.tensor([100.0, 200.0, 300.0])
y = F.softmax(x)
# → tensor([0., 0., 1.])  # 極端な分布に！
```

内積が大きくなると、ソフトマックス後の分布が極端になり、勾配が消失する。

### 4. 解決策：√d でスケーリング

```python
scaled_dot_product = dot_product / np.sqrt(d)
```

**なぜ√d？**
- 標準正規分布のベクトル同士の内積の分散は d
- √d で割ると分散が 1 に正規化される

### 5. スケーリングの効果

| 条件 | 分散 |
|------|------|
| スケーリングなし | ≈ d |
| スケーリングあり | ≈ 1 |

```python
# 視覚化
plt.hist(dot_products, alpha=0.5, label='Without scaling')
plt.hist(scaled_dot_products, alpha=0.5, label='With scaling')
```

## Scaled Dot-Product Attention

```python
scores = torch.matmul(Q, K.t()) / math.sqrt(d_k)
weights = F.softmax(scores, dim=-1)
output = torch.matmul(weights, V)
```

## ポイント

1. **次元が大きいほど問題が深刻**: d=512 など実用的な次元では必須
2. **√d で正規化**: 分散を1に保つ
3. **学習の安定化**: 勾配の大きさが適切に保たれる
