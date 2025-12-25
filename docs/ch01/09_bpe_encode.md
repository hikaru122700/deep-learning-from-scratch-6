# BPEエンコード（事前学習データの準備）

## 学習目標

学習済みBPEトークナイザーを使って、**事前学習用データをエンコード**し、バイナリファイルとして保存する方法を学ぶ。

## 主要概念

### 1. 処理フロー

```
テキストファイル (.txt)
        ↓ BPEエンコード
トークンIDリスト (Python list)
        ↓ numpy配列に変換
numpy配列 (np.uint16)
        ↓ バイナリ保存
バイナリファイル (.bin)
```

### 2. 実装

```python
import numpy as np
from codebot.tokenizer import BPETokenizer

# 事前学習用データのエンコード
text = open("codebot/tiny_codes.txt").read()
tokenizer = BPETokenizer.load_from("codebot/merge_rules.pkl")
ids = tokenizer.encode(text, show_progress=True)  # プログレスバーを表示

# numpy配列に変換して保存
ids_array = np.array(ids, dtype=np.uint16)
ids_array.tofile("codebot/tiny_codes.bin")

print(f"トークンID数: {len(ids_array)}")
print(f"最初の20個のトークンID: {ids_array[:20]}")
```

### 3. データ型の選択

| データ型 | 範囲 | サイズ |
|----------|------|--------|
| `np.uint8` | 0〜255 | 1バイト |
| `np.uint16` | 0〜65,535 | 2バイト |
| `np.uint32` | 0〜4,294,967,295 | 4バイト |

**np.uint16を使用する理由**:
- 語彙サイズ1000なら十分
- GPT-2の語彙サイズ（50,257）でも対応可能
- ファイルサイズを抑えられる

### 4. バイナリファイルの利点

| 形式 | ファイルサイズ | 読み込み速度 |
|------|--------------|-------------|
| テキスト（JSON等） | 大きい | 遅い |
| pickle | 中程度 | 中程度 |
| **バイナリ（.bin）** | **最小** | **最速** |

### 5. バイナリファイルの読み込み

```python
# 後でデータを読み込む場合
ids_array = np.fromfile("codebot/tiny_codes.bin", dtype=np.uint16)
```

## 実行方法

```bash
cd ch01
python 09_bpe_encode.py
```

## 出力

- `codebot/tiny_codes.bin`: エンコード済みデータ
- コンソール出力:
  ```
  トークンID数: 1,234,567
  最初の20個のトークンID: [256 101 102 ...]
  ```

## ファイルサイズの計算

```
ファイルサイズ = トークン数 × 2バイト（uint16）
```

例: 100万トークン → 約2MB

## ポイント

1. **show_progress=True**: 大量データのエンコード時に進捗を確認
2. **uint16**: 語彙サイズが65,535以下なら最も効率的
3. **tofile/fromfile**: NumPyの高速なバイナリI/O

## 次のステップ

このバイナリファイルを使って：
- データローダーの実装（ch03）
- 言語モデルの事前学習（ch03）

## 完成したパイプライン

```
tiny_codes.txt
      ↓ 07_tiny_codes.py（BPE学習）
merge_rules.pkl
      ↓ 09_bpe_encode.py（エンコード）
tiny_codes.bin → 事前学習へ
```
