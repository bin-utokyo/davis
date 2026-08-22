# Davis運営者向け導入ガイド

[English](operator-installation_en.md)

このガイドは，Davisのmetadataを管理し，実データとcatalogをR2へ公開する運営者向けです．データを取得するだけの場合は，[参加者向け導入ガイド](participant-installation.md)を参照してください．

## まず読む5節

通常の運営作業は，次の5節だけで完結します．初めて担当する方は，上から順に読んでください．GitとDavisの詳しい仕組み，例外対応，公開事故からの復旧は後半にまとめています．

1. [Davis CLIのインストール](#1-davis-cliのインストール)
2. [repositoryの準備](#2-repositoryの準備)
3. [運営sessionの準備](#3-運営sessionの準備)
4. [個人作業branchの準備](#4-個人作業branchの準備)
5. [1件のdatasetを更新する標準手順](#5-1件のdatasetを更新する標準手順)

## 1. Davis CLIのインストール

### 必要なもの

- Git
- Davis repositoryへのGitHub access
- 運営者限定で共有された運営共通コード

通常の運営作業にRust，Cargo，Python，DVC，Google Drive，R2秘密鍵は必要ありません．Davis本体を開発・buildする場合だけRustとCargoが必要です．

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

このcommandはDavisの実行fileをGitHub Releasesから取得して，`davis` commandとして使える場所へ配置します．RustやCargoをPCへインストールする処理ではありません．

新しいターミナルを開き，確認します．

```text
davis --version
davis operator --help
```

## 2. repositoryの準備

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

各commandの意味は次のとおりです．

- `git status`: 現在のbranchと未commit変更を表示します．fileは変更しません．`working tree clean`なら未commit変更はありません．
- `git switch main`: 作業対象を公式履歴の基点である`main`へ切り替えます．
- `git pull --ff-only`: GitHubの最新`main`を取得し，履歴が一直線の場合だけlocalを進めます．localとGitHubが分岐している場合は，自動mergeせず停止します．これが「早送り」です．

### VS Codeでrepositoryを準備する場合

初回はCommand Paletteから`Git: Clone`を選び，repository URLを入力し，保存先を選んでclone後のfolderを開きます．既にclone済みの場合は，VS Codeでその`davis` folderを開きます．

左下のbranch名を選択して`main`へ切り替えます．続いてSource Control viewの`…`から`Pull`を選びます．変更fileが表示されている場合や，conflict・mergeを求められた場合は操作を進めず，運営内で相談してください．terminal版の`git pull --ff-only`のほうが，「分岐時に何も変更せず停止する」条件を厳密に保証できます．

## 3. 運営sessionの準備

運営から案内されたDavis WebのURLを指定し，運営共通コードを対話promptへ入力します．コードをcommand引数やshell履歴へ残す必要はありません．

```text
davis operator login <Davis Web URL>
davis operator status
```

運営共通コード自体は保存されず，失効可能な運営sessionだけが端末へ保存されます．通常は有効期間中に再認証する必要はありません．`push`と`publish`には運営sessionを使いますが，`get`と`pull`にはdownload専用の参加者sessionが必要です．運営者が実データを取得する場合は，別途`davis login <Davis Web URL>`を一度実行してください．

VS Codeを使用する場合は，メニューのTerminalから新しいterminalを開き，同じcommandを入力します．sessionはVS CodeではなくPCのuser設定directoryへ保存されるため，通常のterminalとVS Codeのterminalで共通して利用できます．

## 4. 個人作業branchの準備

運営者は`main`ではなく，個人作業branchで更新します．Davisはbranch名のprefixやGitHubユーザー名との一致を要求しません．運営内で識別できる任意の名前を使用できます．同じ個人branchの継続利用を推奨しますが，新しいbranchを作る運用も可能です．

### 初回だけ行う操作

最新`main`から個人作業branchを作ります．

```bash
git status
git switch main
git pull --ff-only
git switch -c <個人作業branch名>
```

- `git switch -c <名前>`: 現在の`main`と同じ内容から新しいbranchを作り，そのbranchへ切り替えます．
- この時点で`git push`する必要はありません．最初の`davis push`がbranchをGitHubへ登録します．

VS Codeでは，左下のbranch名を選択して`main`へ切り替え，Source Control viewの`…`から`Pull`を実行します．もう一度左下のbranch名を選択し，`Create new branch...`から任意の名前を入力します．作成元を尋ねられた場合は`main`を選択します．

### 2回目以降，編集前に行う操作

前回の更新がPull Requestで`main`へmerge済みであることを確認してから，個人branchを最新`main`まで早送りします．

```bash
git status
git switch <個人作業branch名>
git fetch origin
git merge --ff-only origin/main
```

- `git fetch origin`: GitHubの最新履歴を取得しますが，現在のbranchや作業fileは変更しません．
- `git merge --ff-only origin/main`: 個人branchをGitHub上の最新`main`と同じ位置まで進めます．履歴が分岐している場合はmerge commitを作らず停止します．

VS Codeでは，左下のbranch名から個人branchへ切り替え，Source Control viewの`…`から`Fetch`を実行します．その後Command Paletteの`Git: Merge Branch...`で`origin/main`を選択できます．ただし，VS Codeの通常のMerge操作は`--ff-only`を明示しないため，Source Control Graphで一直線に早送りできる状態と確認できる場合だけ使用してください．conflict，merge commit，未commit変更が表示された場合は確定せず停止してください．確実なのは上記terminal commandです．

`davis push`は，名前の付いた`main`以外のbranchで実行できます．detached HEADと`main`からの実行は拒否されます．branch名の形式は問いません．

## 5. 1件のdatasetを更新する標準手順

1. 前節の手順で個人branchへ切り替え，最新`main`まで早送りします．
2. 公開中の実データを持っていない場合だけ，編集前に次を実行します．`pull`は現在公開中のManifestに合わせて実データを取得・更新します．編集後に実行するとlocal変更を上書きするため，編集後には実行しません．

```bash
davis pull routes/Matsuyama
```

3. 取得直後の実データが現在のManifestと一致するか確認する場合は，編集前に次を実行します．`verify`は検査だけを行い，fileを変更しません．編集後はManifestと一致しなくなるため，失敗するのが正常です．

```bash
davis verify routes/Matsuyama
```

4. 担当datasetの実データと`schema.yaml`を編集します．PDFとDatasetManifestは手作業で編集しません．
5. 実際には変更せず，予定だけを確認します．`--dry-run`は新規・変更fileをhashして不足Object数とupload容量を表示しますが，cache，PDF，R2，Gitを変更しません．

```bash
davis push routes/Matsuyama --dry-run
```

6. `Missing objects`，`Existing objects`，`Upload size`を確認します．意図しないfileや容量が表示された場合は通常の`push`へ進みません．
7. 問題がなければ本pushを実行します．`-m`はGitへ記録する変更説明です．省略時は`data: update <dataset>`になります．

```bash
davis push routes/Matsuyama -m "data: update routes/Matsuyama"
```

この1 commandが，ObjectとManifestの作成，不足ObjectのR2 upload，日英PDF生成，対象fileのGit commit，現在の個人branchのGitHubへのpushまで行います．VS Codeでstage，commit，pushを追加実行する必要はありません．Source Control viewがcleanになり，左下の同期表示に未送信commitがなければ完了です．

8. GitHubで個人branchから`main`へのPull Requestを作り，他の運営者にreviewを依頼します．VS CodeにGitHub Pull Requests and Issues拡張機能がある場合は，Source Control viewの`Create Pull Request`から作成できます．拡張機能がなければGitHub Webを使用します．
9. review後，Pull Requestをmerge commitで`main`へmergeします．個人branchを継続利用する場合は削除しません．
10. 公開担当者を1人決め，最新`main`へ切り替えて公開します．

```bash
git switch main
git pull --ff-only
git status
davis publish
```

ここでの`git status`は，公開前に未commit変更がないことを確認するための検査です．`davis publish`も最新`origin/main`とcleanなworking treeを再検査し，条件を満たさなければ公開しません．VS Codeでは左下のbranch名から`main`へ切り替え，Source Control viewの`…`から`Pull`を選べます．その後，VS Codeのterminalで`davis publish`を実行します．

11. `Catalog published: yes`を確認し，Webを強制再読み込みして名称，schema，license，file数，PDF，downloadを確認します．

ここまでが通常作業です．以降は，commandの役割，安全設計，例外時の対応を詳しく知りたい場合に参照してください．

## Git操作とDavis操作の違い

Gitはcode，schema，PDF，DatasetManifestの履歴を管理し，R2は実データのimmutable Objectを保存します．実データはGitへcommitしません．Davisは両者を安全な順序で同期します．

### このガイドで使用するGit command

| command | 目的 | 実データ・R2への影響 |
| --- | --- | --- |
| `git clone <URL>` | repositoryを初めてPCへ複製する | なし |
| `git status` | branchと未commit変更を確認する | なし |
| `git switch <branch>` | `main`と個人作業branchを切り替える | なし |
| `git pull --ff-only` | 現在のbranchをGitHubの状態へ安全に早送りする | なし |
| `git fetch origin` | branchを変更せずGitHubの最新履歴を取得する | なし |
| `git merge --ff-only origin/main` | 個人branchを最新`main`まで安全に早送りする | なし |

通常のデータ更新では，`git add`，`git commit`，`git push`を個別に実行しません．公式運営sessionを使う`davis push`が，対象datasetに限定してこれらを実行します．Git commandだけでは実データを取得・upload・公開できません．

### VS Codeで対応する操作

| Git操作 | VS Codeでの操作 | 通常作業での注意 |
| --- | --- | --- |
| branch確認・切替 | 左下のbranch名を選択 | 変更中fileがある場合は切り替えないでください |
| branch作成 | 左下のbranch名から`Create new branch...` | 作成元は最新`main`にします |
| fetch | Source Control viewの`…`→`Fetch` | 履歴だけを取得し，fileは変更しません |
| pull | Source Control viewの`…`→`Pull` | `main`を最新化するときに使います |
| push | Source Control viewの`…`→`Push` | 通常は`davis push`が自動実行するため不要です |
| pullとpush | `Sync Changes`または左下の同期icon | 両方を連続実行するため，通常のDavis更新手順では使用しません |

VS CodeのSource Control操作とterminalのGit commandは，同じrepository状態を変更します．片方で行った操作はもう片方にも反映されます．

画面名称や詳細は，VS Code公式文書の[Branches and Worktrees](https://code.visualstudio.com/docs/sourcecontrol/branches-worktrees)と[Repositories and Remotes](https://code.visualstudio.com/docs/sourcecontrol/repos-remotes)も参照してください．

### 運営で使用するDavis command

| command | 目的 | 公開Webへの影響 |
| --- | --- | --- |
| `davis --version` | インストール済みversionを確認する | なし |
| `davis update` | 最新releaseと更新方法を確認する | なし |
| `davis login <URL>`・`logout` | download用の参加者sessionを保存・削除する | なし |
| `davis operator login <URL>`・`status`・`logout` | upload・公開用の運営sessionを管理する | なし |
| `davis list` | 利用可能なdatasetを一覧表示する | なし |
| `davis info <dataset>` | file，容量，schema整備状況を確認する | なし |
| `davis get <dataset>` | datasetまたは選択fileを初回取得する | なし |
| `davis pull <dataset>` | dataset全体を公開中Manifestへ同期する | なし |
| `davis pull` | 全datasetを初回取得または同期する | なし |
| `davis verify [dataset]` | local実データを現在のDavis ManifestのBLAKE3と照合する | なし |
| `davis push <dataset> --dry-run` | 更新予定のObjectと容量を確認する | なし |
| `davis push <dataset> [-m <message>]` | 担当datasetを準備し，R2と個人branchへ送る | なし |
| `davis push`・`davis push --all` | 全datasetを検査し，R2と個人branchへ送る | なし |
| `davis publish` | review済みの最新`main`を公開する | あり |

`davis ingest`，`davis documents`，`davis index`は開発・保守用です．通常の更新では使用しません．通常の`push`は前回Manifestとlocal cacheが一致する未変更fileを再利用し，新規・変更・cache欠落fileだけをhashします．全fileを読み直す場合だけ`--rehash`を指定します．

`get`では`--file`を繰り返してfile・directoryを選択でき，`--pdf-ja`と`--pdf-en`で説明PDFを追加できます．schemaは標準で保存され，`--no-schema`を指定した場合だけ省略します．`pull`にも同じ文書optionがあります．完全な引数一覧は`davis <command> --help`で確認してください．

### 正本と自動生成物

個人作業者が編集するのは，担当datasetの実データと`schema.yaml`です．Davisはdataset rootにある実fileを直接検出するため，`.dvc`は使用しません．fileの追加，変更，名前変更，移動，削除は，次の`push`でDatasetManifestへ反映されます．

日英PDFとDatasetManifestは派生物です．通常の`davis push`は，R2 upload成功後に，変更されたschemaまたはBLAKE3 Object IDに関係するPDFだけを決定的に生成します．個人作業者はPDFやManifestを手作業で編集せず，`git add`，`git commit`，`git push`も個別に実行しません．

通常の`davis push`は次を順に行います．

1. 個人branchと`origin/main`の状態を検査する
2. 対象dataset外の未commit変更がないことを検査する
3. 未変更fileを前回Manifestとlocal cacheから再利用し，必要なfileのBLAKE3 ObjectとDatasetManifestを生成する
4. 不足ObjectをR2へuploadする
5. upload成功後，変更されたschemaまたはObject IDに関係する日英PDFだけを生成する
6. 対象datasetのschema，PDF，Manifestだけをstageする
7. commitし，現在の個人branchをGitHubへpushする

途中で失敗した場合は，後続処理へ進みません．R2へ保存済みのObjectは内容address形式でimmutableなため，Git処理が失敗しても既存公開を壊しません．公開Catalogは`davis publish`まで変わりません．

### なぜ公開作業を1人ずつ行うのか

CatalogIndexはDavis全体の現在状態を表し，`catalog/current.json`はそのrevisionを1つだけ指します．2人が別々のbranchや古い`main`から同時に公開すると，後から完了したほうが先の公開内容を上書きし，別datasetの更新をWeb上から消す可能性があります．R2の実データObjectは内容address形式で保存されるため通常は失われませんが，公開catalogから参照されなくなります．

本番公開時は，次を守ってください．

- 公開担当者を1人に限定する
- 最新の`main`だけから公開する
- working treeがcleanであることを確認する
- 対象Pull Requestがmerge済みであることを確認する
- 別の公開作業が終わるまで次の公開を始めない
- 1件だけを同期する場合はdataset IDを指定し，全件同期では無印`davis push`または互換aliasの`davis push --all`を使用する

複数datasetのPull Requestをほぼ同時に進める場合，担当者はそれぞれの個人branchからObjectを先に同期できます．すべてのObject同期とreviewが完了した後にPull Requestをmergeし，公開担当者が最新`main`から一度だけ`davis publish`を実行できます．

### `push`と`publish`を分離する理由

複数人運用では，次の2操作を分けることが重要です．

```text
davis push <dataset>   # 個人branchでPDF・Manifest・R2・Gitを同期
davis publish          # review・merge後，最新mainからCatalogIndexだけを公開
```

Objectのuploadは内容address形式であり，同じ内容は重複uploadされず，Catalogから参照されるまで参加者には現れません．そのため個人branchから先に実行しても安全にできます．一方，`publish`は参加者が見る状態を変更するため，最新`main`と公開担当者の確認が必要です．

Davisはこの2操作を分離しています．`davis push`は個人branchでも利用できますが，`davis publish`はreview済みの最新`main`専用です．公開担当者はR2秘密鍵を持つ必要がなく，運営sessionで公開できます．

### 誤って公開した場合

保護を無効化した旧CLI等により，個人branchや古い`main`から公開したことに気づいた場合は，R2 Objectを削除しないでください．現在のCLIはこの操作を拒否しますが，まず運営内へ共有し，正しい最新`main`を持つ1台から対象datasetを検証して再公開します．

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

## 公式環境以外のstorageを利用する場合

このガイドの個人branch，`origin/main`，Pull Requestに関する規則は，Davis公式Catalogを複数人で安全に更新するための運用方針です．運営sessionを使わず，`.davis/config.toml`でfilesystemまたはS3互換remoteへ直接接続する`davis push`では，公式branch名やGitHubを要求せず，Git commit・pushも自動実行しません．別組織で利用する場合は，ObjectとManifestの共通形式を保ったまま，その組織のreview・公開方針を定めてください．

## 困ったとき

- `davis operator status`が期限切れを示す場合は，`davis operator login <URL>`を再実行してください．
- `davis`が見つからない場合は，新しいターミナルを開いてください．
- 旧版が呼ばれる場合は，macOS・Linuxでは`which -a davis`，Windowsでは`Get-Command davis -All`で確認してください．
- Gitに未commit変更がある状態で移行に失敗した場合は，強制的に戻さず運営内で相談してください．
