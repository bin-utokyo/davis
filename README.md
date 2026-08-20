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

Rust版CLIの正式な実行file名は`davis`です．旧Python版を残す必要がある場合は，`davis-legacy`という名前へ退避できます．

```bash
cargo install --path crates/davis-cli --locked --root ~/.local
davis --help
```

旧Python版から切り替えた後は，通常の操作には`davis`を使用してください．

参加者は，運営から案内されたDavis WebのURLを指定して一度loginします．招待codeは画面に表示されない対話promptから入力し，sessionが有効な間は再入力不要です．以後の`list`，`info`，`get`は公開catalogと認証済みDownload APIを利用します．`get`はcurrent directory以下に`data/<Dataset root>/...`を再現し，取得済みObjectはlocal cacheから再利用します．

```bash
davis login https://<配布されたURL>
davis list
davis info network/matsuyama
davis get network/matsuyama
davis logout
```

Webカタログの「CLIコマンドをコピー」で得られるcommandには`--service-url`が含まれます．CLI側で未loginの場合は，任意のdirectoryから実行してもその場で招待codeを入力でき，login後に同じ処理のままdownloadを続行します．Web browserのlogin sessionとCLIのlogin sessionは別です．

非対話環境では，招待codeをcommand line引数やshell履歴へ残さず，標準入力から渡します．

```bash
printf '%s\n' "$DAVIS_INVITE_CODE" | davis login https://<配布されたURL> --invite-code-stdin
```

sessionはOSごとのuser設定directoryに権限を限定して保存します．これはMVPの暫定credential storageであり，release配布前にOS credential store adapterへ置き換えられる境界を維持します．operatorのR2 credentialとは独立しており，参加者sessionにはupload，削除，Object一覧の権限がありません．

運営者は，参加者codeとは別の運営共通codeを使用して一度loginします．運営sessionは既定30日で，期限内は再入力不要です．期限切れ後の`push`では対話的にcodeを再入力してsessionを更新できます．運営sessionはObjectのuploadとcatalog公開だけに使用され，R2秘密鍵を各端末へ配布する必要はありません．

```bash
davis operator login https://davis-web.davis-bin.workers.dev
davis operator status
davis push network/matsuyama --dry-run
davis push network/matsuyama
davis operator logout
```

大容量Objectは32 MiBごとのmultipart uploadとしてR2へ送信します．運営共通code自体は端末へ保存せず，失効可能な運営sessionだけを権限を限定して保存します．

現行DVC metadataと実データの整合性確認，BLAKE3 objectの生成，DatasetManifestの更新には次を使用します．

```bash
cargo run -p davis-cli -- verify
cargo run -p davis-cli -- ingest --all
```

R2またはfilesystem remoteへの差分uploadには`davis-next push`を使用します．1 Datasetだけを指定できるほか，`davis-next push --all --dry-run`で全Datasetの差分を安全に確認できます．通常の`push`は不足Objectのupload後に最新の`schema.yaml`からCatalogIndexを生成し，revision単位で保存してから`catalog/current.json`を切り替えます．Objectの差分がなくschemaだけが変わった場合もWebへ反映されます．設定例は`.davis/config.example.toml`にあります．

対話terminalでは，`push`のlocal検証・uploadと，remoteを指定した`get`のdownloadについて，処理済みObject数，処理済み容量，割合，残り時間をprogress barで表示します．pipeやCI等の非対話実行ではprogress barを自動的に非表示にし，既存の標準出力を維持します．

## Webカタログ

WebはR2上の現在のCatalogIndexを読みます．まだ一度もCatalogIndexがR2へ公開されていない環境では，deploymentに同梱した静的indexへfallbackします．開発時に静的indexを更新する場合は，すべての`schema.yaml`から次のように生成します．

```bash
cargo run -p davis-cli -- index
cd web/davis-web
pnpm install
pnpm dev
```

Webカタログでは，名称・説明・地域・年・形式・license・schema状態・列情報による検索，Dataset・File詳細，Raw YAML表示，複数選択，合計容量表示，対応する`davis get` commandのcopyができます．Workerには，共通招待code，失効可能なsession，短寿命Download Grant，private R2 Object配信の共通APIがあります．CLIとWeb UIの両方がこのAPIへ接続し，Webでは利用条件を確認して選択fileを個別にdownloadできます．

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
