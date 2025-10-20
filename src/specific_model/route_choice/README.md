# RouteChoice モデル Docker 実行手順

## 概要
このプロジェクトは経路選択モデル（RL/割引率RL等）の推定・シミュレーションをDocker環境で実行するものです。

## ディレクトリ構成
```
code/           # Pythonコード(main_rl.py等)
data/input/     # 入力データ
data/output/    # 出力結果
requirements.txt
DockerFile
docker-compose.yml
.env            # 実行モードやパスの設定
```

## 事前準備
1. 必要なファイル（データ・コード・.env）を配置してください。
2. Python依存パッケージは`requirements.txt`で管理しています。

## .envファイル例
```ini
NETWORK_FROM_OSM=true
MAPMATCHING=true
ESTIMATE=true
SIMULATE=true
ASSIGNMENT=true

INPUT=./input
OUTPUT=./output
MODEL_NAME=RL
TRANSPORTATION_MODE=300

POLYGON_COORD='[[139.698544, 35.660225], [139.698544, 35.656913], [139.705410, 35.656913], [139.705410, 35.660225]]'
```

## ビルドと実行
```sh
# イメージのビルド
$ docker-compose build

# コンテナの起動（.envの設定に従いmain_rl.pyが実行されます）
$ docker-compose up
```

## コマンド・モード切替
- `.env`の`NETWORK_FROM_OSM`, `MAPMATCHING`, `ESTIMATE`, `SIMULATE`を`true`/その他の単語に変更することで、実行内容を切り替えられます。
- 必要に応じて`INPUT`や`OUTPUT`のパスも変更してください。
- `TRANSPORTATION_MODE`はPPデータのFeederデータに合わせてください。

## 注意事項
- データや出力の永続化のため、`volumes`でローカルディレクトリとコンテナ内ディレクトリをバインドしています。
- Pythonコードの修正は`code/`ディレクトリ内で行ってください。
- 
---

# JupyterNotebook 実行手順

code/main.ipynb内の指示に従って順にセルを実行して下さい．
