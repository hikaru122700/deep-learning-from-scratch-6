# Tiny Codes BPE学習（実践）

## 学習目標

実際のコードデータセット（**Tiny Codes**）を使って、BPEトークナイザーを学習する方法を学ぶ。

## 主要概念

### 1. ファイル構成

```
codebot/
├── tiny_codes.txt      # 学習用コードデータ
├── tokenizer.py        # BPETokenizerクラス
└── merge_rules.pkl     # 学習済みマージルール（出力）
```

### 2. 学習スクリプト

```python
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import pickle
from codebot.tokenizer import train_bpe

vocab_size = 1000  # 語彙数
text = open("codebot/tiny_codes.txt").read()
merge_rules = train_bpe(text, vocab_size)

# 学習済みマージルールをファイルに保存
with open("codebot/merge_rules.pkl", "wb") as f:
    pickle.dump(merge_rules, f)
```

### 3. 作業ディレクトリの設定

```python
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')
```

- スクリプトの場所からプロジェクトルートに移動
- `codebot.tokenizer` をインポート可能にする

### 4. pickleによるシリアライズ

```python
# 保存
with open("codebot/merge_rules.pkl", "wb") as f:
    pickle.dump(merge_rules, f)

# 読み込み（後で使用）
with open("codebot/merge_rules.pkl", "rb") as f:
    merge_rules = pickle.load(f)
```

## 語彙サイズの選び方

| 語彙サイズ | 特徴 |
|-----------|------|
| 小さい（1000程度） | 学習が速い、圧縮率は低い |
| 中程度（10000程度） | バランスが良い |
| 大きい（50000以上） | 圧縮率が高い、学習に時間がかかる |

## 実行方法

```bash
cd ch01
python 07_tiny_codes.py
```

## 出力

- `codebot/merge_rules.pkl`: 学習済みマージルール
  - 辞書形式: `{(token1, token2): new_id, ...}`
  - vocab_size - 256 - 1 個のルール

## ポイント

1. **データセット**: コードに特化したデータで学習することで、プログラミング言語に適したトークン化が可能
2. **pickle形式**: Pythonオブジェクトをそのまま保存/復元
3. **再利用性**: 一度学習すれば、同じマージルールを何度でも使用可能

## 次のステップ

学習したトークナイザーを使って：
- トークンの確認（08_eval.py）
- テキストのエンコード（09_bpe_encode.py）
