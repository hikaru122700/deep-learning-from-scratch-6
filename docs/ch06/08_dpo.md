# DPO（Direct Preference Optimization）

## 学習目標

人間の選好データから直接学習する**DPO**を理解する。RLHFの代替手法。

## 主要概念

### 1. DPOデータセット

```python
class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer, context_len):
        with open(data_path) as f:
            data = json.load(f)

        for item in data:
            sample = self._create_sample(
                item['prompt'],
                item['chosen'],    # 好まれる応答
                item['rejected']   # 好まれない応答
            )
            self.samples.append(sample)

    def _create_sample(self, prompt, chosen, rejected):
        prompt_ids = self.tokenizer.encode(prompt)
        chosen_ids = prompt_ids + self.tokenizer.encode(chosen)
        rejected_ids = prompt_ids + self.tokenizer.encode(rejected)

        # マスク: プロンプト部分は0、応答部分は1
        chosen_mask = [0] * len(prompt_ids) + [1] * (len(chosen_ids) - len(prompt_ids))
        rejected_mask = [0] * len(prompt_ids) + [1] * (len(rejected_ids) - len(prompt_ids))

        return chosen_ids, chosen_mask, rejected_ids, rejected_mask
```

### 2. シーケンスの対数確率

```python
def get_sequence_logprobs(model, ids, mask):
    logits = model(ids)
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    labels = ids[:, 1:]

    per_token_logprobs = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    # マスクを適用（応答部分のみ）
    masked_logprobs = per_token_logprobs * mask[:, 1:]
    return masked_logprobs.sum(dim=-1)
```

### 3. DPO損失関数

```python
def compute_dpo_loss(model, ref_model, chosen_ids, chosen_mask,
                     rejected_ids, rejected_mask, beta):
    # 現在のモデルのlog-prob
    chosen_logprobs = get_sequence_logprobs(model, chosen_ids, chosen_mask)
    rejected_logprobs = get_sequence_logprobs(model, rejected_ids, rejected_mask)

    # 参照モデルのlog-prob（勾配なし）
    with torch.no_grad():
        ref_chosen_logprobs = get_sequence_logprobs(ref_model, chosen_ids, chosen_mask)
        ref_rejected_logprobs = get_sequence_logprobs(ref_model, rejected_ids, rejected_mask)

    # DPO loss
    logits = beta * (
        (chosen_logprobs - rejected_logprobs) -
        (ref_chosen_logprobs - ref_rejected_logprobs)
    )
    return -F.logsigmoid(logits).mean()
```

### 4. 学習ループ

```python
model = GPT.load_from(pretrain_model_path, device=device)
ref_model = GPT.load_from(pretrain_model_path, device=device)
ref_model.eval()  # 参照モデルは固定

for i in pbar:
    chosen_ids, chosen_mask, rejected_ids, rejected_mask = next(data_iter)

    loss = compute_dpo_loss(
        model, ref_model,
        chosen_ids, chosen_mask,
        rejected_ids, rejected_mask,
        beta
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## ハイパーパラメータ

```python
batch_size = 8
learning_rate = 5e-6
beta = 0.1  # KLペナルティの強さ
max_iters = 1000
```

## DPOの損失関数の直感

```
損失 = -log(σ(β × (モデルの選好差 - 参照モデルの選好差)))

モデルの選好差 = log P(chosen) - log P(rejected)
```

- モデルが chosen を rejected より好むほど損失が下がる
- 参照モデルからの逸脱にペナルティ

## RLHF vs DPO

| 項目 | RLHF | DPO |
|------|------|-----|
| 報酬モデル | 必要 | 不要 |
| 強化学習 | 必要（PPO等） | 不要 |
| 実装複雑度 | 高 | 低 |
| 安定性 | やや不安定 | 安定 |
