# GRPO（Group Relative Policy Optimization）

## 学習目標

報酬ベースの強化学習手法**GRPO**を使ったモデル改善を理解する。

## 主要概念

### 1. GRPOデータセット

```python
class GRPODataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        # 足し算タスク
        for i in range(1, 10):
            for j in range(1, 10):
                prompt = f"### Instruction:\n{i}+{j}=\n\n### Response:\n"
                ground_truth = i + j
                self.data.append((prompt, ground_truth))
```

### 2. 報酬関数

```python
def calculate_reward(ground_truth, response):
    try:
        matches = re.findall(r'(-?\d+)', response)
        if matches:
            predicted = int(matches[-1])
            return 1.0 if predicted == ground_truth else 0.0
        return 0.0
    except:
        return 0.0
```

正解なら報酬1、不正解なら報酬0。

### 3. グループ生成とアドバンテージ計算

```python
def generate_group(model, tokenizer, prompts, gts, group_size):
    all_prompts, all_responses, all_advantages = [], [], []

    for prompt, gt in zip(prompts, gts):
        responses = []
        for _ in range(group_size):
            full_text = generate(model, tokenizer, prompt, temperature=1.0)
            response = full_text[len(prompt):]
            responses.append(response)

        # 報酬を計算
        rewards = torch.tensor([calculate_reward(gt, r) for r in responses])
        # アドバンテージ = 報酬 - グループ平均
        advantages = rewards - rewards.mean()

        for response, advantage in zip(responses, advantages):
            all_prompts.append(prompt)
            all_responses.append(response)
            all_advantages.append(advantage)

    return all_prompts, all_responses, torch.stack(all_advantages)
```

### 4. GRPO損失関数

```python
def grpo_loss(model, old_model, ids, mask, advantages, epsilon=0.2):
    # 現在モデルの確率
    probs = compute_probs(model, ids)
    # 古いモデルの確率
    with torch.no_grad():
        old_probs = compute_probs(old_model, ids)

    # 確率比
    ratio = probs / old_probs
    advantages = advantages.unsqueeze(-1)

    # クリッピング
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

    mask = mask[:, 1:]
    token_objective = torch.min(unclipped, clipped) * mask
    return -token_objective.sum() / mask.sum()
```

### 5. 学習ループ

```python
for i in pbar:
    prompts, gts = next(data_iter)

    # 古いモデルを更新
    old_model.load_state_dict(model.state_dict())

    # グループ生成
    all_prompts, all_responses, all_advantages = generate_group(
        old_model, tokenizer, prompts, gts, group_size
    )

    # バッチ作成
    ids, mask = dataset.get_batch(all_prompts, all_responses, device)

    # 複数回更新
    for _ in range(n_update_per_generation):
        optimizer.zero_grad()
        loss = grpo_loss(model, old_model, ids, mask, all_advantages, epsilon)
        loss.backward()
        optimizer.step()
```

## ハイパーパラメータ

```python
learning_rate = 2e-6
max_iters = 200
n_update_per_generation = 2
epsilon = 0.2  # クリッピング範囲
group_size = 8
batch_size = 32
```

## GRPOのポイント

1. **グループ相対**: 絶対的な報酬ではなくグループ内での相対評価
2. **クリッピング**: 大きな更新を防ぎ安定した学習
3. **複数回更新**: 同じ生成データで複数回パラメータ更新
4. **古いモデル保持**: 確率比計算のため

## PPO vs GRPO

| 項目 | PPO | GRPO |
|------|-----|------|
| ベースライン | Valueネットワーク | グループ平均 |
| 追加モデル | Critic必要 | 不要 |
| 実装 | 複雑 | シンプル |
