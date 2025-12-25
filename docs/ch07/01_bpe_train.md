# BPE学習（大規模データ）

## 学習目標

**大規模データセット**（OpenWebText等）でBPEトークナイザーを学習する方法を理解する。

## 主要概念

### 1. 大規模学習の設定

```python
vocab_size = 50000  # GPT-2と同等の語彙サイズ
file_path = "webbot/owt_train.txt"
```

### 2. 並列BPE学習

```python
if __name__ == '__main__':
    merge_rules = train_bpe(
        file_path,
        vocab_size,
        num_processes=8,
        num_chunks=64
    )

    with open("webbot/merge_rules.pkl", "wb") as f:
        pickle.dump(merge_rules, f)
```

### 3. train_bpe関数（ch04で実装済み）

主な機能：
- ファイルをチャンク分割
- 並列で事前トークン化
- キャッシュを使った効率的なマージ

## 設定の比較

| プロジェクト | データ | 語彙サイズ |
|------------|--------|-----------|
| CodeBot | tiny_codes.txt | 1,000 |
| StoryBot | tiny_stories_train.txt | 10,000 |
| **WebBot** | owt_train.txt | 50,000 |

## 大規模学習のポイント

### 語彙サイズの選び方

| サイズ | 特徴 |
|--------|------|
| 1,000 | 小規模実験向け |
| 10,000 | 小〜中規模データ |
| 32,000 | LLaMA |
| 50,000 | GPT-2 |
| 100,000+ | 多言語対応 |

### 計算時間の目安

```
語彙サイズ 50,000、データ数GB の場合：
- 事前トークン化: 数分〜十数分
- マージ学習: 数十分〜数時間
```

### メモリ使用量

- チャンク処理によりメモリ使用量を制限
- 事前トークンのカウント辞書がボトルネック

## 使用方法

```bash
cd ch07
python 01_bpe_train.py
```

出力：
- `webbot/merge_rules.pkl`: 学習済みマージルール

## ポイント

1. **`if __name__ == '__main__'`**: Windowsでのマルチプロセス対応
2. **num_processes=8**: CPU コア数に合わせて調整
3. **num_chunks=64**: ファイルサイズに応じて調整
