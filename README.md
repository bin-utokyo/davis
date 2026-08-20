# Davis

Davisは，交通データの取得から行動モデルの実行までを，一つの流れにつなぐためのplatformです．現在は，既存データカタログの機能を維持しながら，内容アドレス型storage，Rust CLI，静的Webカタログの基盤を開発しています．

## 現在の構成

```text
crates/
  davis-core/       # DatasetManifest，内容アドレス，ローカルcache
  davis-catalog/    # DVC・schema.yaml読込み，静的index生成
  davis-storage/    # filesystem・S3互換storage
  davis-cli/        # list，info，index，verify，ingest，push，get

web/
  davis-web/        # schema検索・絞り込み・複数選択Webカタログ

packages/
  dataset_cli/      # 既存Python CLI

src/
  specific_model/  # 行動モデルの推定・simulation code
  base_model/      # 交通simulation code
```

## Rust CLIの開発

RustとCargoを導入した環境では，repository rootから次のように実行できます．

```bash
cargo run -p davis-cli -- list
cargo run -p davis-cli -- info network/matsuyama
cargo run -p davis-cli -- get network/matsuyama
cargo run -p davis-cli -- get network/matsuyama --file link.csv
```

現行DVC metadataと実データの整合性確認，BLAKE3 objectの生成，DatasetManifestの更新には次を使用します．

```bash
cargo run -p davis-cli -- verify
cargo run -p davis-cli -- ingest --all
```

R2またはfilesystem remoteへの差分uploadには`davis push`を使用します．設定例は`.davis/config.example.toml`にあります．

## Webカタログ

Webが使用する静的indexは，すべての`schema.yaml`から生成します．

```bash
cargo run -p davis-cli -- index
cd web/davis-web
pnpm install
pnpm dev
```

Webカタログでは，名称・説明・地域・年・形式・license・schema状態・列情報による検索，Dataset・File詳細，Raw YAML表示，複数選択，合計容量表示，対応する`davis get` commandのcopyができます．

実データのWeb download，共通招待code，署名付きURLはR2・Worker接続後に追加します．

## 検証

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
uv run pytest -q

cd web/davis-web
pnpm test
pnpm lint
```

## Document

- [Davis仕様書](docs/davis-spec.md)
- [Platform構想](docs/davis-platform-concept.md)
- [既存dataset CLI](packages/dataset_cli/README.md)
- [Base model](src/base_model/README.md)

質問やcontributionについては，行動モデル夏の学校の運営までお問い合わせください．
