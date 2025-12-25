# LLM-as-a-Judge（モデル評価）

## 学習目標

**LLM-as-a-Judge**を使ってモデルの品質を評価する方法を理解する。

## 主要概念

### 1. LLM-as-a-Judgeとは

大規模言語モデル（GPT-4など）を審査員として使い、生成テキストを評価する手法。

### 2. 評価関数

```python
def evaluate_story(client, story):
    evaluation_prompt = f"""以下の子供向けストーリーを2つの観点で1-5点で評価してください。

ストーリー:
{story}

評価観点:
1. Coherence（一貫性）: 論理的につながっているか、物語として筋が通っているか
2. Grammar（文法）: 文法的に正しい英語か

以下のJSON形式で回答してください:
{{
    "coherence": <1-5の整数>,
    "grammar": <1-5の整数>,
    "comment": "<評価の簡単な理由>"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": evaluation_prompt}],
        max_tokens=300,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
```

### 3. 複数イテレーションの比較

```python
model_paths = {
    500: 'storybot/model_iter_500.pt',
    5000: 'storybot/model_iter_5000.pt',
    40000: 'storybot/model_pretrain.pt',
}

results = {}
for iteration, model_path in model_paths.items():
    model = GPT.load_from(model_path, device=device)
    iteration_results = []

    for i in range(num_samples):
        story = generate(model, tokenizer, prompt, max_new_tokens, temperature)
        scores = evaluate_story(client, story)
        iteration_results.append({"story": story, "scores": scores})

    results[iteration] = iteration_results
```

### 4. 結果のサマリー

```python
for iteration in model_paths.keys():
    scores_list = [r["scores"] for r in results[iteration]]

    print(f"\nIteration {iteration}:")
    for key in ["coherence", "grammar"]:
        values = [s[key] for s in scores_list]
        avg = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        print(f"  {key}: {avg:.2f} ± {std:.2f}")
```

## 評価観点の例

| 観点 | 説明 |
|------|------|
| Coherence | 論理的一貫性 |
| Grammar | 文法の正確さ |
| Creativity | 創造性 |
| Engagement | 面白さ |

## 期待される結果

学習が進むにつれて：
- 初期（500イテレーション）：低品質
- 中期（5000イテレーション）：改善
- 終盤（40000イテレーション）：高品質

## ポイント

1. **JSON形式**: 構造化された評価を取得
2. **response_format**: パース処理がシンプルに
3. **統計的評価**: 複数サンプルの平均と標準偏差
4. **API料金**: GPT-4o-miniで低コスト評価
