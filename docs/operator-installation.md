# Davis運営者向け導入ガイド

[English](operator-installation_en.md)

このガイドは，Davisのmetadataを管理し，実データとcatalogをR2へ公開する運営者向けです．データを取得するだけの場合は，[参加者向け導入ガイド](participant-installation.md)を参照してください．

## 必要なもの

- Git
- Davis repositoryへのGitHub access
- 運営者限定で共有された運営共通コード

通常の運営作業にRust，Cargo，Python，DVC，Google Drive，R2秘密鍵は必要ありません．Davis本体を開発・buildする場合だけRustとCargoが必要です．

## Davis CLIのインストール

### macOS・Linux

```bash
curl --proto '=https' --tlsv1.2 -fsSL https://raw.githubusercontent.com/bin-utokyo/davis/main/scripts/install.sh | sh
```

### Windows

PowerShellで実行します．

```powershell
irm https://raw.githubusercontent.com/bin-utokyo/davis/main/scripts/install.ps1 | iex
```

installerはOSとCPUを自動判定し，macOS・Windows・Linuxの対応する実行ファイルを導入します．旧Python版が`uv`で導入されている場合は，旧版を削除してから`davis`を新版へ置き換えます．

新しいターミナルを開き，確認します．

```text
davis --version
davis operator --help
```

## repositoryの準備

初めて導入する場合は，repositoryをcloneします．

```bash
git clone https://github.com/bin-utokyo/davis.git
cd davis
```

既にclone済みで，まだデータ更新作業を始めていない場合は，作業treeがcleanであることを確認してから現行`main`へ更新します．

```bash
git status
git switch main
git pull --ff-only
```

`git status`に未commit変更が表示された場合は，削除，stash，強制更新をせず，作業内容を確認してから運営内で相談してください．

## 参加者ログイン

運営者がCLIからデータを取得する場合は，参加者共通コードでもloginします．

```text
davis login https://davis-web.davis-bin.workers.dev
```

参加者sessionはdownload専用です．

## 運営者ログイン

```text
davis operator login https://davis-web.davis-bin.workers.dev
davis operator status
```

`Operator code:`に，運営者限定で案内された運営共通コードを入力します．コード自体は保存されず，権限を限定した運営sessionが保存されます．sessionは既定30日間有効で，期限内は何度pushしても再入力不要です．期限切れ時は，次の`push`中にコードを一度入力するとsessionが更新されます．

## 接続確認

repository rootで，担当datasetを指定してdry runします．

```text
davis push routes/Matsuyama --dry-run
```

`Remote: ... (operator session)`と表示され，Object数とupload予定容量が確認できれば準備完了です．`--dry-run`ではR2と公開catalogを変更しません．

## 日常作業の原則

### GitとDavisの役割

運営作業では，名前が似ている2種類の`push`を区別してください．

| 操作 | 送信先 | 主な対象 | 公開Webへの影響 |
| --- | --- | --- | --- |
| `git push` | GitHub | code，`.dvc`，`schema.yaml`，説明PDF，DatasetManifest | Pull Requestを作るだけでは公開Webは変わりません |
| `davis push` | R2 | 担当datasetのimmutableな実データObject | 公開Webは変わりません |
| `davis publish` | Davis Web API | review済み`main`から生成したCatalogIndex | 公開Webのcatalogを切り替えます |

Gitをmetadataと説明資料の正本，R2を実データのObject storageとして扱います．`schema.yaml`と日英PDFはGitだけに保存し，R2へ重複保存しません．CatalogIndexには検索に必要なYAML内容とPDFのGitHub参照が含まれます．

### 個人作業branchの原則

運営者はそれぞれ個人作業branchで編集します．`main`へ直接commitせず，次の状態から作業を始めます．

```bash
git status
git switch main
git pull --ff-only
git switch -c <個人作業branch名>
```

既にその人の継続作業branchがある場合は，新しいbranchを更新のたびに増やさず，運営内の方針に従って同じ作業branchを継続利用して構いません．ただし，新しい作業へ入る前に`main`との差分と未commit変更を確認してください．

個人作業branchでは，次を行えます．

- 担当する実データを取得・編集する
- `.dvc`，`schema.yaml`，日英PDF，DatasetManifestを更新する
- `davis verify`を実行する
- `davis push <dataset> --dry-run`で予定差分を確認する
- Gitへcommitし，`git push`してPull Requestを作る

個人作業branchから`davis push`を実行して構いません．このcommandは内容address形式のObjectだけをR2へ同期し，公開Catalogを変更しません．ただし，対象datasetを明示し，先に`--dry-run`で対象と容量を確認してください．公開状態を変更する`davis publish`は個人作業branchから実行できません．CLIがbranch，working tree，`origin/main`との一致を検査して拒否します．

### 1件のdatasetを更新する標準手順

1. `main`を更新し，個人作業branchへ移動します．
2. 担当datasetの実データ，`.dvc`，`schema.yaml`を更新します．
3. YAMLから日英PDFを生成し，YAMLとPDFを同じGit commitへ含めます．
4. 対象datasetだけを検証します．

```bash
davis verify routes/Matsuyama
davis push routes/Matsuyama --dry-run
git status
```

5. 意図しないdatasetやfileが差分へ含まれていないこと，予定upload容量，schemaとPDFの対応を確認します．
6. 個人作業branchから担当datasetのObjectをR2へ同期します．この時点では参加者向けWebは変わりません．

```bash
davis push routes/Matsuyama
```

7. `Objects synchronized: yes`と`Catalog published: no`を確認します．`davis push`が更新したDatasetManifestを含め，metadataと説明資料をcommitし，GitHubへ`git push`してPull Requestを作ります．実データそのものはGitへ追加しません．
8. 他の運営者が，列定義，利用条件，対象年，file名，削除・移動，DatasetManifest，予定容量をreviewします．
9. Pull Requestを`main`へmergeします．
10. 公開担当者を1人決め，他の公開作業が進行中でないことを運営内で確認します．公開担当者は実データをローカルに持つ必要はありません．必要なObjectは手順6で既にR2へ同期されています．
11. 公開担当者の端末で最新`main`へ移動します．

```bash
git switch main
git pull --ff-only
git status
```

12. `git status`がcleanであることを確認してから，公開します．`davis publish`自身も`main`，cleanなworking tree，`origin/main`との完全一致を検査します．Catalogが参照するObjectがR2に不足している場合も公開せず終了します．

```bash
davis publish
```

13. `Catalog published: yes`を確認し，Webを強制再読み込みして名称，schema，license，file数，downloadを確認します．

### なぜ公開作業を1人ずつ行うのか

CatalogIndexはDavis全体の現在状態を表し，`catalog/current.json`はそのrevisionを1つだけ指します．2人が別々のbranchや古い`main`から同時に公開すると，後から完了したほうが先の公開内容を上書きし，別datasetの更新をWeb上から消す可能性があります．R2の実データObjectは内容address形式で保存されるため通常は失われませんが，公開catalogから参照されなくなります．

本番公開時は，次を守ってください．

- 公開担当者を1人に限定する
- 最新の`main`だけから公開する
- working treeがcleanであることを確認する
- 対象Pull Requestがmerge済みであることを確認する
- 別の公開作業が終わるまで次の公開を始めない
- Object同期では通常は担当datasetだけを指定し，`davis push --all`は全体検査や明示的な一括同期に限定する

複数datasetのPull Requestをほぼ同時に進める場合，担当者はそれぞれの個人branchからObjectを先に同期できます．すべてのObject同期とreviewが完了した後にPull Requestをmergeし，公開担当者が最新`main`から一度だけ`davis publish`を実行できます．

### `push`と`publish`を分離する理由

複数人運用では，次の2操作を分けることが重要です．

```text
davis push <dataset>   # 個人branchからimmutable ObjectだけをR2へ同期
davis publish          # review・merge後，最新mainからCatalogIndexだけを公開
```

Objectのuploadは内容address形式であり，同じ内容は重複uploadされず，Catalogから参照されるまで参加者には現れません．そのため個人branchから先に実行しても安全にできます．一方，`publish`は参加者が見る状態を変更するため，最新`main`と公開担当者の確認が必要です．

Davisはこの2操作を分離しています．`davis push`は個人branchでも利用できますが，`davis publish`はreview済みの最新`main`専用です．公開担当者はR2秘密鍵を持つ必要がなく，運営sessionで公開できます．

### 誤って公開した場合

個人branchや古い`main`から公開したことに気づいた場合は，R2 Objectを削除しないでください．まず運営内へ共有し，正しい最新`main`を持つ1台から対象datasetを検証して再公開します．

```bash
git switch main
git pull --ff-only
git status
davis publish
```

誤ったmetadata自体が`main`へmerge済みの場合は，Git上で修正またはrevertするPull Requestをreview・mergeしてから再公開します．R2 Objectの削除は復旧操作ではありません．参照されていないObjectの整理は，保持期間と削除承認の手順に従う別の保守作業として行います．

### credentialと秘密情報

- 運営共通コードをrepository，commit，issue，Pull Request，terminalのcommand引数，メールの公開宛先へ記載しないでください．
- `.davis`内のsession情報をGitへ追加しないでください．
- 運営共通コード流出時は，共通コードと運営access revisionを差し替え，既存sessionを一括失効します．
- R2秘密鍵を通常の運営端末へ配布しません．

データfileの追加，移動，名称変更，削除は，Catalog上のIDや既存利用者の再現性にも影響します．通常の内容更新と同じ扱いで済ませず，Pull Requestに変更理由と影響範囲を記載してreviewしてください．導入確認だけを目的としてObjectをuploadしたり，Catalogを公開したりしないでください．

## 更新

CLIは24時間に1回だけ最新版を確認し，新しいreleaseがある場合は通常commandの完了後に案内します．`davis update`を実行すると，現在のversionと最新版を比較し，OSに対応する更新commandを表示します．表示されたinstallerを実行すると最新版へ更新できます．repository，実データ，login session，運営sessionは維持されます．

## 困ったとき

- `davis operator status`が期限切れを示す場合は，`davis operator login <URL>`を再実行してください．
- `davis`が見つからない場合は，新しいターミナルを開いてください．
- 旧版が呼ばれる場合は，macOS・Linuxでは`which -a davis`，Windowsでは`Get-Command davis -All`で確認してください．
- Gitに未commit変更がある状態で移行に失敗した場合は，強制的に戻さず運営内で相談してください．
