# Custom Fine-tuning PARSeq for Soccer Player Jersey Number Recognition

## 1. Dataset Preparation

### 1.1 Installing Required Dependencies

```bash
cd parseq
poetry install
```

### 1.2 Creating LMDB Datasets

```bash
cd parseq

# 訓練データとテストデータを作成し、./dataディレクトリに出力
python tools/create_lmdb_from_torso_crops.py \
  --train_images ~/soccernet/dataset/torso_crops/train_bbox \
  --train_csv_file ~/soccernet/dataset/torso_crops/train_bbox_minimax.csv \
  --test_images ~/soccernet/dataset/torso_crops/test_bbox \
  --test_csv_file ~/soccernet/dataset/torso_crops/test_bbox_minimax.csv \
  --output_dir ./data
```

### 1.3 Directory Structure

```
./data/
├── train/
│   ├── data.mdb
│   └── lock.mdb
├── val/    # 検証データ
│   ├── data.mdb
│   └── lock.mdb
└── test/   # テストデータ
    ├── data.mdb
    └── lock.mdb
```

## 2. Configuration File Preparation

### 2.1 Character Set Configuration

背番号認識では、0〜9 の数字のみを使用します。これを以下のファイルで設定します

```bash
# ./configs/charset/10_digits.yaml
charset_train: '0123456789'
charset_test: '0123456789'
```

### 2.2 Dataset Configuration

データセットのパスやその他の設定を以下のファイルで構成します

```bash
# ./configs/dataset/jersey.yaml

data:
  root_dir: ./data
  train_dir: train
  remove_whitespace: false
  normalize_unicode: false
  augment: true  # データ拡張を有効化

model:
  max_label_length: 2  # 背番号は最大2桁
  batch_size: 32  # バッチサイズを小さくしてエポックあたりのバッチ数を増やす
  weight_decay: 0.0

trainer:
  max_epochs: 30
  devices: 1  # 利用可能なGPU数に応じて調整
  val_check_interval: 5  # トレーニングバッチ数に合わせて調整
```

## 3. Running Custom Fine-tuning

事前学習済みの PARSeq モデルのエンコーダー部分を再利用し、デコーダーと分類ヘッドを背番号認識用に再トレーニングするカスタムファインチューニングプロセスを実行します：

```bash
python custom_finetune.py charset=10_digits dataset=jersey
```
