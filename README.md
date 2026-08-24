# Davis

[English](READMEen.md)

Davisは，交通データの取得から行動モデルの実行までを，一つの流れにつなぐためのplatformです．現在は，既存データカタログの機能を維持しながら，内容アドレス型storage，Rust CLI，静的Webカタログの基盤を開発しています．

## 導入

通常利用ではRust，Cargo，Pythonは不要です．GitHub Releaseで配布するOS・CPU別の実行ファイルをinstallerが自動選択します．利用目的に応じて，次のガイドを参照してください．

- [参加者向け導入ガイド](docs/participant-installation.md) ([English](docs/participant-installation_en.md))
- [運営者向け導入ガイド](docs/operator-installation.md) ([English](docs/operator-installation_en.md))

運営者は，運営者向け導入ガイド冒頭の「まず読む5節」だけで，インストール，repository・session・個人branchの準備，dataset更新，review，公開まで進められます．terminalとVS Codeの操作を併記しています．

Davis本体の開発・buildを行う場合だけ，RustとCargoが必要です．

CLIは，通常commandの完了後に24時間に1回だけ最新版を確認します．更新情報の正本はGitHub Releaseへ添付する`latest-version.json`であり，CLI releaseのたびにWebを再deployする必要はありません．新しいreleaseがある場合は，通常処理やJSON出力を妨げずに更新案内を表示します．`davis update`を実行すると，更新内容を表示してinstallするか`y/N`で確認し，承認後にOS対応installerを実行します．確認を省く場合は`davis update --yes`を使用します．

## 現在の構成

```text
crates/
  davis-core/       # DatasetManifest，内容アドレス，ローカルcache
  davis-catalog/    # Davis Manifest・schema.yaml読込み，静的index生成
  davis-document/   # schemaから決定的な日英PDFを生成
  davis-storage/    # filesystem・S3互換storage
  davis-cli/        # list，info，index，verify，ingest，push，get

web/
  davis-web/        # schema検索・絞り込み・複数選択Webカタログ

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

Rust版CLIの正式な実行file名は`davis`です．

旧Python CLIは現行releaseへ同梱しません．調査や復元が必要な場合は，[`legacy-python-final` tag](https://github.com/bin-utokyo/davis/tree/legacy-python-final/packages/dataset_cli)または`legacy/python-cli-v0` branchを参照できます．

```bash
cargo install --path crates/davis-cli --locked --root ~/.local
davis --help
```

参加者は，運営から案内されたDavis WebのURLを指定して一度loginします．招待codeは画面に表示されない対話promptから入力し，sessionが有効な間は再入力不要です．以後の`list`，`info`，`get`は公開catalogと認証済みDownload APIを利用します．`get`はcurrent directory以下に`data/<Dataset root>/...`を再現し，取得済みObjectはlocal cacheから再利用します．

```bash
davis login https://<配布されたURL>
davis list
davis info network/matsuyama
davis get network/matsuyama
davis pull network/matsuyama
davis logout
```

`get`は初回取得とfile単位の選択取得，`pull`はdataset全体の初回取得または現在のManifestへの同期に使用します．`pull`は既存fileをremoteの内容で更新するため，local編集を残したまま実行しないでください．

dataset IDを省略した無印`davis pull`と`davis push`は，全datasetを対象にします．dataset IDを指定すると，その1件だけを対象にします．`davis push --all`も互換性のため同じ全件操作として使用できます．

Webカタログの「CLIコマンドをコピー」で得られるcommandには`--service-url`が含まれます．CLI側で未loginの場合は，任意のdirectoryから実行してもその場で招待codeを入力でき，login後に同じ処理のままdownloadを続行します．Web browserのlogin sessionとCLIのlogin sessionは別です．

非対話環境では，招待codeをcommand line引数やshell履歴へ残さず，標準入力から渡します．

```bash
printf '%s\n' "$DAVIS_INVITE_CODE" | davis login https://<配布されたURL> --invite-code-stdin
```

sessionはOSごとのuser設定directoryに権限を限定して保存します．これはMVPの暫定credential storageであり，release配布前にOS credential store adapterへ置き換えられる境界を維持します．operatorのR2 credentialとは独立しており，参加者sessionにはupload，削除，Object一覧の権限がありません．

運営者は，参加者codeとは別の運営共通codeを使用して一度loginします．運営sessionは既定30日で，期限内は再入力不要です．期限切れ後の`push`または`publish`では対話的にcodeを再入力してsessionを更新できます．運営sessionはObjectのuploadとcatalog公開だけに使用され，R2秘密鍵を各端末へ配布する必要はありません．

```bash
davis operator login https://davis-web.davis-bin.workers.dev
davis operator status
davis push network/matsuyama --dry-run
davis push network/matsuyama
davis publish
davis operator logout
```

大容量Objectは32 MiBごとのmultipart uploadとしてR2へ送信します．運営共通code自体は端末へ保存せず，失効可能な運営sessionだけを権限を限定して保存します．

現在のDavis Manifestと実データのBLAKE3整合性確認には次を使用します．

```bash
cargo run -p davis-cli -- verify
cargo run -p davis-cli -- ingest --all
```

公式operator sessionを使い，`main`以外の個人作業branchから`davis push <dataset>`を実行すると，新規・変更・cache欠落fileのBLAKE3を計算し，未変更fileは前回Manifestとlocal cacheから再利用します．branch名の形式はDavisでは指定しません．不足ObjectのR2 uploadに成功した後，変更されたschemaまたはObject IDに関係する日英PDFだけを生成し，対象datasetのschema，PDF，Manifestだけをstage・commitして，現在の個人branchをGitHubへpushします．公開Catalogは変更しません．事前確認には`--dry-run`を使用します．dry runはrepository，cache，R2，Gitを変更しません．全fileを読み直す場合だけ`--rehash`を指定します．DVCと`.dvc`は通常経路で使用しません．

各fileのManifestには，そのObject IDへ変わった`davis push`の日を`updated_at`として記録します．Object IDが変わらなければ日付も維持され，datasetの最終更新日は構成fileの最も新しい日付から導出されます．これらはWeb Catalogのdataset一覧，file選択，詳細画面に表示されます．

operator sessionを使わず，`.davis/config.toml`でfilesystemまたはS3互換remoteへ直接接続する場合，`davis push`はObjectとlocal Manifestを同期しますが，公式branch名，`origin/main`，GitHubを要求せず，Git commit・pushも行いません．これにより別組織のrepository，MinIO，S3，local storageでも同じManifestとObject形式を利用できます．

review済みmetadataをWebへ反映するときは，Pull Requestをmergeした後，最新かつcleanな`main`から`davis publish`を実行します．このcommandは`main`，`origin/main`との一致，working tree，R2 Objectの網羅性を検査してから，CatalogIndexをrevision単位で保存し，`catalog/current.json`を切り替えます．個人作業branchではObjectを先に同期できますが，Catalogは公開できません．設定例は`.davis/config.example.toml`にあります．

対話terminalでは，`push`のlocal検証・uploadと，remoteを指定した`get`のdownloadについて，処理済みObject数，処理済み容量，割合，残り時間をprogress barで表示します．pipeやCI等の非対話実行ではprogress barを自動的に非表示にし，既存の標準出力を維持します．

`davis get`と`davis pull`は，実データと対応する`schema.yaml`を標準で保存します．日本語・英語の説明PDFは任意で追加でき，schemaを保存しない場合は明示的なoptionが必要です．

```bash
davis get routes/Matsuyama
davis get routes/Matsuyama --pdf-ja
davis get routes/Matsuyama --pdf-en
davis get routes/Matsuyama --pdf-ja --pdf-en
davis get routes/Matsuyama --no-schema
davis pull routes/Matsuyama
davis pull routes/Matsuyama --pdf-ja --pdf-en
davis pull
```

`get`の取得先に選択対象の既存fileがある場合は，上書きするか確認します．`y`で一括上書き，`N`またはEnterで無変更のまま中止します．自動実行など確認を省く場合は`--force`を指定します．非対話環境では，意図しない上書きを避けるため`--force`なしでは停止します．

`schema.yaml`と説明PDFはGitを正本とし，R2 Objectとしては重複保存しません．`schema.yaml`の内容とPDFのGitHub URLはCatalogIndexへ記録されます．CLIとWebはYAMLをCatalog APIから保存し，PDFをGitHubから取得します．実データだけがprivate R2 Objectとして配信されます．

## Webカタログ

WebはR2上の現在のCatalogIndexを読みます．まだ一度もCatalogIndexがR2へ公開されていない環境では，deploymentに同梱した静的indexへfallbackします．開発時に静的indexを更新する場合は，すべての`schema.yaml`から次のように生成します．

```bash
cargo run -p davis-cli -- index
cd web/davis-web
pnpm install
pnpm dev
```

Webカタログでは，名称・説明・地域・年・形式・license・schema状態・列情報による検索，Dataset・File詳細，Raw YAML表示，複数選択，合計容量表示，対応する`davis get` commandのcopyができます．download確認画面では，`schema.yaml`を初期選択し，日本語PDFと英語PDFを任意で追加できます．schemaの選択を外すと，将来の整形・推定機能との接続に再取得が必要になる可能性を画面内で警告します．Workerには，共通招待code，失効可能なsession，短寿命Download Grant，private R2 Object配信の共通APIがあります．

公式Workerは，有効なDownload GrantでObject配信endpointへ到達した回数を「download試行数」としてCloudflare Analytics Engineへ記録します．File ID，path，Object ID，full／rangeの区別だけを保存し，個人情報やsession tokenは保存しません．統計記録の失敗はdownloadを失敗させません．公開API，CLI，Manifest，schemaの形式は変更しません．

## 検証

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

cd web/davis-web
pnpm test
pnpm lint
```

## ライセンス

Davisのソフトウェア本体は，[MIT License](LICENSE)のもとで公開されています．データセットにはこのソフトウェアライセンスは適用されません．各データファイルの利用条件は，対応する`schema.yaml`の`license`を確認してください．

## Document

- [参加者向け導入ガイド](docs/participant-installation.md) ([English](docs/participant-installation_en.md))
- [運営者向け導入ガイド](docs/operator-installation.md) ([English](docs/operator-installation_en.md))
- [Davis仕様書](docs/davis-spec.md)
- [Platform構想](docs/davis-platform-concept.md)
- [Base model](src/base_model/README.md)

## 文書更新方針

利用者に読んでもらうREADME，導入ガイド，運用ガイドを変更する場合は，日本語版と英語版を同じcommitまたはPull Requestで同時に更新します．一方だけを先行更新しません．日本語版と英語版の対応関係は，各文書の冒頭または上位READMEから相互に確認できる状態を維持します．

質問やcontributionについては，行動モデル夏の学校の運営までお問い合わせください．
