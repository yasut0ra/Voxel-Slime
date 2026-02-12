# Voxel Slime

Physarum に着想を得た 3D 生成アートです。複数種のエージェントがボクセル空間でトレイルを残し、餌場・毒場・捕食者との相互作用で有機的なネットワークを成長させます。

## 生成サンプル（GIF）

| fungi | coral | nebula |
| --- | --- | --- |
| ![fungi-mip](outputs/fungi/fungi-mip.gif) | ![coral-mip](outputs/coral_obj/coral-mip.gif) | ![nebula-mip](outputs/nebula/nebula-mip.gif) |

## クイックスタート

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

まずは見栄えのよい標準シミュレーション（`fungi` プリセット、2,000 ステップ）:

```bash
python run.py --preset fungi --seed 7 --out outputs/fungi
```

捕食者なしで比較したい場合:

```bash
python run.py --preset fungi --seed 7 --no-predator --out outputs/fungi_no_pred
```

最大値投影フレームから GIF を作成:

```bash
python scripts/make_gif.py \
  --frames-dir outputs/fungi/frames/mip \
  --pattern "mip_*.png" \
  --out outputs/fungi/fungi_mip.gif \
  --fps 24
```

トレイル場（全種チャネル合成）の等値面から OBJ メッシュを書き出し:

```bash
python run.py --preset coral --steps 1200 --seed 42 --export-obj --obj-threshold 22 --out outputs/coral_obj
```

## CLI オプション

```bash
python run.py \
  --preset {fungi,nebula,coral} \
  --size 96 \
  --agents 50000 \
  --steps 2000 \
  --seed 42 \
  --species 3 \
  --interaction-mode {symmetric,cyclic} \
  --self-attract 1.0 \
  --cross-attract -0.35 \
  --predator-ratio 0.14 \
  --predator-attract 1.8 \
  --predator-repel -1.25 \
  --predator-consume-amount 4.0 \
  --predator-capture-radius 1 \
  --food-weight 1.25 \
  --toxin-weight 1.3 \
  --food-sources 3 \
  --toxin-sources 2 \
  --field-source-speed 0.42 \
  --field-source-jitter 0.05 \
  --field-injection-radius 2 \
  --food-injection-amount 22 \
  --toxin-injection-amount 18 \
  --field-diffuse-rate 0.34 \
  --field-evap-rate 0.06 \
  --out outputs/run_name \
  --save-every 10 \
  --render-colormap magma \
  --render-gamma 0.9 \
  --env-overlay-strength 0.28 \
  --export-obj \
  --obj-threshold 9.5 \
  --boundary {wrap,reflect} \
  --slice-axis {x,y,z} \
  --slice-index 48
```

## アルゴリズム（5ステップ）

1. 3D ボクセルグリッド内に、多数のエージェントをランダムな位置・向き・種別で初期化します。
2. 餌場と毒場の「移動ソース」がトレイルとは別の外部場を形成し、毎ステップ注入・拡散・蒸発します。
3. 各エージェントは進行方向の周辺（前方、左右、上下、対角）で、全種トレイル + 餌場 + 毒場をセンシングします。
4. 種間相互作用行列でスコア化して旋回・移動し、各自の種チャネルへトレイルを堆積します。
5. 捕食者種は prey trail を追跡し、捕捉半径内の prey をリスポーンさせつつ trail を削り、崩壊と再生のダイナミクスを生みます。

## 出力

- `outputs/<run>/frames/slice/slice_XXXXXX.png`: 軸方向スライス画像（カラー、自動コントラスト調整）。
- `outputs/<run>/frames/mip/mip_XXXXXX.png`: 最大値投影画像（カラー、自動コントラスト調整）。
- `outputs/<run>/mesh.obj`（任意）: 頂点法線付きの marching cubes メッシュ。

## プリセット

- `fungi`: 3種 + 捕食者 + 餌/毒場 -> 菌糸状ネットワークに断裂と再接続が混ざる。
- `nebula`: 4種 + 積極的な移動ソース -> 雲状の重なりが脈動する。
- `coral`: 2種 + 強い捕食圧 -> 太い枝が削られながら再成長する。

## 補足

- `--seed` を指定すると再現可能な（決定論的な）実行になります。
- Physarum の particle/trail モデルと、走化性に類似した振る舞いに着想を得ています。
- 試行を高速化したい場合は `--steps`、`--agents`、`--size` を下げてください。
- 見た目を変えるには `--predator-ratio`、`--food-weight`、`--toxin-weight`、`--field-source-speed`、`--render-gamma` の調整が効きます。

## リポジトリ構成

```text
voxel-slime/
  run.py
  src/
    __init__.py
    config.py
    agents.py
    trail.py
    simulate.py
    render.py
    export.py
    utils.py
  configs/
    fungi.yaml
    nebula.yaml
    coral.yaml
  scripts/
    make_gif.py
  requirements.txt
  README.md
  LICENSE
```
