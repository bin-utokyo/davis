# ベースモデル（MFD-RL+Hongo） 実行手順

## 概要
このプロジェクトはMFD-RL+Hongoシミュレーションの実行を行うものです。設定ファイルやデータを用いて、シミュレーションを実施します。

## ディレクトリ構成
```
base_model/
    Hongo/              # Hongo関連の入出力・ソース
    MFDRL-Hongo/
        code/           # Pythonコード(main.py, mfdrlpy.py, hongopy.py等)
        data/           # 入力データ（csv, geojson, shapefile等）
        config/         # 設定ファイル（conf_*.json）
        run.sh          # 実行スクリプト
        requirements.txt
        DockerFile
docker-compose.yml
.env            # 実行モードやパスの設定
```

## 事前準備
1. conf.json（config/配下）でファイルパス等の設定を環境に合わせて修正してください。
2. .envファイルを作成し、必要な環境変数を設定してください。

## 実行方法
1. 初期化→シミュレーションの2段階実行が必要です。
   - .envで `INITIALIZE=true` → データの初期化
   - .envで `INITIALIZE=false` → シミュレーションの実行
2. MFDのパラメータ（mfd_params3.csv）やRLのパラメータ（EstimationResult.csv）は初期化操作で生成されないため、他プログラムの推定値を利用してください。
3. `docker-compose up` でコンテナを起動し、シミュレーションを実行します。

## フォルダ構造例
```
Hongo/
    build/
    src/
MFDRL-Hongo/
    code/
        main.py
        mfdrlpy.py
        hongopy.py
        create_input_mfdrl.py
        create_input_hongo.py
        Utility.py
    data/
        500mesh/
            ehime500m.csv
        Hongo/
            input/
            output/
        MFD-RL/
            input/
                Activity_params.csv
                mfd_params3.csv
                Route_params.csv
            output/
        mobaku/
        output/
    config/
        conf_matsuyama.json
    MFD-RL/
        input/
        output/
```

## 注意事項
- データや出力の永続化のため、必要に応じてディレクトリのバインドやパス設定を行ってください。
