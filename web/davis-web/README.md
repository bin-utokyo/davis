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

現段階では，検索，facet絞り込み，Dataset・File詳細，Raw YAML表示，複数選択，合計容量表示，`davis get` commandのcopyに対応しています．実データのWeb downloadと参加者認証はR2・Worker接続後に追加します．
