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
	specific_model/       # 個別モデル
		mode_choice/        # モード選択モデル（MNL等）
		route_choice/       # 経路選択モデル（RL等）

	base_model/           # ベースモデル（MFD-RL+Hongo）
		Hongo/              # Hongoシミュレーターのソース
		MFDRL-Hongo/        # MFD-RL+Hongoシミュレーターのソース
```

## ダウンロード
最新版のコードはGitHubからクローンしてください：
```sh
git clone https://github.com/bin-utokyo/davis.git
```

## インストール方法

このプロジェクトのCLIツール (`davis-cli`) は、GitHubの[リリースページ](https://github.com/bin-utokyo/davis/releases)からインストールできます。

### 1. リリースページからアセットをダウンロード

最新のリリースから、お使いの環境に合った`whl`ファイルまたは`tar.gz`ファイルをダウンロードします。

### 2. `uv` または `pip` を使ってインストール

ダウンロードしたファイルを指定して、以下のコマンドでインストールします。

**`whl`ファイルの場合:**
```sh
# uv を使う場合
uv pip install /path/to/downloaded_file.whl

# pip を使う場合
pip install /path/to/downloaded_file.whl
```

**`tar.gz`ファイルの場合:**
```sh
# uv を使う場合
uv pip install /path/to/downloaded_file.tar.gz

# pip を使う場合
pip install /path/to/downloaded_file.tar.gz
```

## 注意事項
- 詳細や.envの例は各モデルのREADMEをご参照ください．

## ドキュメント
- データセット管理ツールの詳細は `packages/dataset_cli/README.md` を参照してください．
- ベースモデルの詳細は `src/base_model/README.md` を参照してください．
- 各行動モデルの詳細は `src/specific_model/{モデル名}/README.md` を参照してください．

---

質問やコントリビューションは，行動モデル夏の学校の運営までお問い合わせください．
