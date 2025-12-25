# LLM-as-a-Judge（ペアワイズ比較）

## 学習目標

2つのモデルを**ペアワイズ比較**で評価する方法を理解する。DPO学習の効果を検証。

## 主要概念

### 1. 比較関数

```python
def compare_stories(client, story_a, story_b):
    evaluation_prompt = f"""以下の2つの子供向けストーリーを比較し、どちらがよりハッピーエンドかを判定してください。

【Story A】
{story_a}

【Story B】
{story_b}

どちらがより明るく幸せな結末か、または希望に満ちた内容かを判断してください。
JSON形式で回答: {{"winner": "A" or "B" or "tie", "reason": "簡潔な理由"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": evaluation_prompt}],
        max_tokens=150,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
```

### 2. 位置バイアスの回避

```python
import random

# ランダムに順序を入れ替え
if random.random() < 0.5:
    story_a, story_b = story_pretrain, story_dpo
    mapping = {"A": "pretrain", "B": "dpo"}
else:
    story_a, story_b = story_dpo, story_pretrain
    mapping = {"A": "dpo", "B": "pretrain"}

judgment = compare_stories(client, story_a, story_b)

# 実際の勝者を特定
winner_label = judgment["winner"]
if winner_label == "tie":
    winner = "tie"
else:
    winner = mapping[winner_label]
```

### 3. 比較ループ

```python
model_pretrain = GPT.load_from(model_paths['pretrain'], device=device)
model_dpo = GPT.load_from(model_paths['dpo'], device=device)

wins = {"pretrain": 0, "dpo": 0, "tie": 0}

for i in range(num_comparisons):
    # 両モデルでストーリーを生成
    story_pretrain = generate(model_pretrain, tokenizer, prompt, max_new_tokens, temperature)
    story_dpo = generate(model_dpo, tokenizer, prompt, max_new_tokens, temperature)

    # 位置をランダム化して比較
    # ...

    wins[winner] += 1
```

### 4. 結果のサマリー

```python
print("📊 PAIRWISE COMPARISON RESULTS")

total = num_comparisons
print(f"Pretrain wins: {wins['pretrain']:3d} ({wins['pretrain']/total*100:5.1f}%)")
print(f"DPO wins:      {wins['dpo']:3d} ({wins['dpo']/total*100:5.1f}%)")
print(f"Ties:          {wins['tie']:3d} ({wins['tie']/total*100:5.1f}%)")

# 勝率（tieを除く）
if wins['pretrain'] + wins['dpo'] > 0:
    dpo_winrate = wins['dpo'] / (wins['pretrain'] + wins['dpo']) * 100
    print(f"DPO win rate (excluding ties): {dpo_winrate:.1f}%")
```

## 評価設計のポイント

| 項目 | 対策 |
|------|------|
| 位置バイアス | ランダムに順序を入れ替え |
| サンプル数 | 100回程度の比較 |
| 評価基準 | 明確な観点（ハッピーエンド等） |
| 統計的有意性 | 勝率と信頼区間 |

## 期待される結果

DPOで「ハッピーエンド」を学習した場合：
- DPO勝率 > 50%
- Pretrain勝率 < 50%

## ポイント

1. **位置バイアス**: AとBの提示順序による偏り
2. **同一プロンプト**: 公平な比較のため同じプロンプトを使用
3. **tie**: 明確な差がない場合の選択肢
4. **統計的評価**: 単発ではなく多数の比較
