# BPEエンコード（大規模データ）

## 学習目標

大規模データセットを**並列エンコード**し、学習用バイナリファイルを作成する方法を理解する。

## 主要概念

### 1. エンコード設定

```python
tokenizer = BPETokenizer.load_from("webbot/merge_rules.pkl")
```

### 2. ファイルエンコード

```python
if __name__ == '__main__':
    tokenizer.encode_file(
        "webbot/owt_train.txt",
        "webbot/owt_train.bin",
        num_processes=8
    )

    tokenizer.encode_file(
        "webbot/owt_valid.txt",
        "webbot/owt_valid.bin",
        num_processes=8
    )
```

### 3. encode_file関数の処理フロー

```
入力: owt_train.txt (数GB)
    ↓ チャンク分割
[chunk1, chunk2, ...]
    ↓ 並列エンコード（各プロセス）
[cache1.npy, cache2.npy, ...]
    ↓ memmap結合
owt_train.bin
    ↓ キャッシュ削除
完了
```

## 出力ファイル

| ファイル | 用途 | 形式 |
|----------|------|------|
| owt_train.bin | 学習用 | uint16配列 |
| owt_valid.bin | 検証用 | uint16配列 |

## サイズの目安

```
元テキスト: 10GB
    ↓ BPEエンコード（圧縮率3倍）
トークン数: 約35億
    ↓ uint16保存
出力ファイル: 約7GB
```

## 使用方法

```bash
cd ch07
python 02_bpe_encode.py
```

## 学習での使用

```python
# memmapで読み込み
train_data = np.memmap("webbot/owt_train.bin", dtype=np.uint16, mode='r')
val_data = np.memmap("webbot/owt_valid.bin", dtype=np.uint16, mode='r')

# バッチ取得
x = train_data[idx:idx+context_len]
y = train_data[idx+1:idx+context_len+1]
```

## ポイント

1. **Train/Valid分割**: 別々にエンコード
2. **memmap**: 大規模ファイルを効率的に扱う
3. **並列処理**: マルチコアで高速化
4. **キャッシュ管理**: 一時ファイルは自動削除

## 全体のパイプライン

```
raw text
    ↓ ch07/01_bpe_train.py
merge_rules.pkl
    ↓ ch07/02_bpe_encode.py
train.bin, valid.bin
    ↓ ch06/05_pretrain.py
model_pretrain.pt
```
