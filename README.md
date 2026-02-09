# Pair Correspondence Visualizer

## 1) これは何か
- `main.py pair`: 画像ペア（target/observed/mask）から対応推定と可視化を実行
- `main.py batch`: `data/ViT` 形式のディレクトリを一括処理
- DINO/CLIP/Ours/SIFT/ORB のシナリオを切り替え可能

## 2) 前提（OS, GPU/CPU, 主要依存）
- 想定OS: Windows/Linux
- Python: 3.10+
- 主要依存: `torch`, `transformers`, `opencv-contrib-python`, `Pillow`, `scikit-learn`, `matplotlib`
- カメラ入力を使う場合のみ `pypylon` が必要

## 3) セットアップ（最短手順）
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## 4) 最小実行例（コピペ可能）
```bash
python main.py pair --scenario orb --target data/samples/target.png --observed data/samples/DG/{DGtype}.png --mask data/samples/mask.png --out-dir outputs/demo
```

設定ファイルを使う場合:
```bash
python main.py pair --config configs/default.yaml
```

一括処理:
```bash
python main.py batch --vit-root ../data/ViT --scenarios dino,clip,ours,sift,orb --out-root outputs/batch
```

## 5) データの置き場所（サンプル、取得方法）
- 最小サンプル: `data/samples/target.png`, `data/samples/DG/{DGtype}.png`, `data/samples/mask.png`
- 実データ一括処理は `--vit-root` で指定
- 出力は `--out-dir` / `--out-root` 配下に保存

## 6) よくあるエラーと対処
- `cv2.SIFT_create is unavailable`
  - `opencv-contrib-python` を再インストール
- `Target/Observed image not found`
  - パスが相対か絶対かを確認（コマンド実行位置はこのディレクトリ）
- `pypylon is required`
  - `--use-camera` を使う場合のみ `pip install pypylon`

## 7) プロジェクト構造（各ディレクトリの役割）
```text
pair_correspondence_visualizer/
├─ main.py                 # CLIエントリ
├─ config.py               # シナリオ定義
├─ configs/default.yaml    # 実行設定例
├─ src/                    # 推定・可視化ロジック
├─ data/samples/           # 最小サンプル入力
└─ scripts/smoke_test.py   # スモークテスト
```

## 8) ライセンス/引用（third_party含む）
- このディレクトリには外部リポジトリのソースコード同梱はありません
- 使用モデル・ライブラリのライセンスは各配布元に従ってください
