# Davis プロジェクト

## 概要
本リポジトリは，行動モデル夏の学校で使用可能なデータの配布ツールおよび行動モデルのコードサンプルを提供します：
- データセット管理ツール：データの情報表示，ダウンロード
- 個別モデル：行動モデルの推定・シミュレーションコード
- ベースモデル：ミクロ交通シミュレータ

## 構成
```
packages/
  dataset_cli/          # データセット管理ツール
src/
	specific_model/
		mode_choice/        # モード選択モデル（MNL等）
		route_choice/       # 経路選択モデル（RL等）
	base_model/           # ベースモデル（MFD-RL+Hongo）
		Hongo/              # Hongoシミュレーターのソース
		MFDRL-Hongo/        # MFD-RL+Hongoシミュレーターのソース
```
各モデルディレクトリには以下が含まれます：
- code/           # Pythonコード（main_mc.py, main_rl.py など）
- requirements.txt
- DockerFile
- docker-compose.yml
- .env            # 実行モードやパスの設定
- README.md       # モデルごとの説明

## ダウンロード
最新版のコードはGitHubからクローンしてください：
```sh
git clone https://github.com/bin-utokyo/davis.git
```

## 注意事項
- 詳細や.envの例は各モデルのREADMEをご参照ください．

## ドキュメント
- データセット管理ツールの詳細は `packages/dataset_cli/README.md` を参照してください．
- ベースモデルの詳細は `src/base_model/README.md` を参照してください．
- 各行動モデルの詳細は `src/specific_model/{モデル名}/README.md` を参照してください．

---

質問やコントリビューションは，行動モデル夏の学校の運営までお問い合わせください．
