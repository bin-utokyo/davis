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

- `main`を最新にしてから個人作業branchを作ります．
- 担当dataset以外の実データを取得する必要はありません．
- schemaとmetadataはGitでreviewし，実データはDavisを通じてR2へ送ります．
- 本番の`davis push`はWeb catalogを更新するため，dry runとreviewを済ませてから実行します．
- 運営共通コードをrepository，issue，Pull Request，メールの公開宛先へ記載しないでください．
- コード流出時は，共通コードと運営access revisionを差し替え，既存sessionを一括失効します．

データファイルの追加・移動・削除を含む更新手順は，DatasetManifest中心のworkflowが確定するまで運営内の更新手順に従ってください．導入確認だけを目的として本番`push`を実行しないでください．

## 更新

同じinstallerを再実行すると最新版へ更新できます．repository，実データ，login session，運営sessionは維持されます．

## 困ったとき

- `davis operator status`が期限切れを示す場合は，`davis operator login <URL>`を再実行してください．
- `davis`が見つからない場合は，新しいターミナルを開いてください．
- 旧版が呼ばれる場合は，macOS・Linuxでは`which -a davis`，Windowsでは`Get-Command davis -All`で確認してください．
- Gitに未commit変更がある状態で移行に失敗した場合は，強制的に戻さず運営内で相談してください．
