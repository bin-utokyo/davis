# davis-web

`davis-web`は，DavisのFileSchemaから生成した静的indexを読み込む交通データカタログです．R2や認証が未接続の状態でも，ローカルで検索UIを開発・確認できます．

## Catalog index

repository rootで次を実行すると，現在の`data/**/*.dvc`と`schema.yaml`からindexを再生成します．

```bash
cargo run -p davis-cli -- index
```

生成物は`public/catalog`に保存されます．Git上のDVC metadataとFileSchemaが正本であり，JSON indexは派生物です．

## Local development

Node.js 22以降とpnpmを使用します．

```bash
pnpm install
pnpm dev
```

開発画面は通常`http://localhost:3000`で開きます．

## Verification

```bash
pnpm test
pnpm lint
```

現段階では，検索，facet絞り込み，Dataset・File詳細，Raw YAML表示，複数選択，合計容量表示，`davis get` commandのcopyに対応しています．参加者認証とR2 downloadのWorker APIへCLIとWeb UIの両方を接続済みです．Web UIは共通招待codeをHttpOnly cookieへ交換し，利用条件の確認後に選択fileを個別downloadします．

## Download API

Workerには，CLIとWebが共用するversion 1 APIがあります．Catalog metadataは公開したまま，実データだけを認証で保護します．

| Endpoint | 用途 |
| --- | --- |
| `POST /api/v1/auth/exchange` | 共通招待codeをbrowser cookieまたはCLI tokenへ交換 |
| `GET /api/v1/auth/session` | sessionの有効性と期限を確認 |
| `POST /api/v1/auth/logout` | browser sessionを削除 |
| `POST /api/v1/download-grants` | File ID集合を5分有効のdownload URLへ交換 |
| `GET /api/v1/download?grant=...` | private R2 Objectをstreaming download |

Download APIは公開CatalogのFile IDだけを受理し，任意のR2 key指定，Object一覧，PUT，DELETEを提供しません．Range requestにも対応します．

local開発では`.dev.vars.example`を`.dev.vars`へcopyし，実際の値へ置き換えます．`DAVIS_TOKEN_SECRET`には32文字以上のrandom値を使用してください．productionでは値をsourceや通常の環境変数へ保存せず，Cloudflare Worker Secretとして登録します．

```bash
pnpm exec wrangler secret put DAVIS_INVITE_CODE
pnpm exec wrangler secret put DAVIS_OPERATOR_CODE
pnpm exec wrangler secret put DAVIS_TOKEN_SECRET
```

Cloudflareへloginし，Secretを登録した後は，次のcommandでbuild済みassetとWorkerを`davis-bin` accountへdeployします．既存の`davis-bmss`を`DAVIS_DATA`としてbindingし，R2 Objectのuploadや削除は行いません．

```bash
pnpm deploy
```

R2 binding名は`DAVIS_DATA`，既定bucket名は`davis-bmss`です．`DAVIS_ACCESS_REVISION`を変更して再deployすると，旧招待codeで発行済みのsessionとdownload grantを一括失効できます．sessionは既定30日で最大180日，download grantは既定5分で最大15分です．

運営者認証は`DAVIS_OPERATOR_CODE`と`DAVIS_OPERATOR_ACCESS_REVISION`を使用し，参加者認証から分離します．運営sessionは既定30日で最大90日です．CLIは認証済みAPIを通じてR2 multipart uploadとcatalog公開を行うため，運営者のPCへR2秘密鍵を配布する必要はありません．運営code流出時は`DAVIS_OPERATOR_CODE`と`DAVIS_OPERATOR_ACCESS_REVISION`を変更して再deployすると，既存の運営sessionを一括失効できます．

CLIからは，deployment URLと共通招待codeを使用します．

```bash
davis login https://<deployment URL>
davis list
davis get network/matsuyama
```

CLIは`catalog/datasets.json`と`catalog/files.json`からDatasetManifestを再構成し，`POST /api/v1/download-grants`で不足Objectだけを取得します．したがって，参加者はrepositoryのclone，DVC，R2 credentialを必要としません．
