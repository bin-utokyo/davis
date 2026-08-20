# Davis参加者向け導入ガイド

[English](participant-installation_en.md)

このガイドは，Davis WebまたはDavis CLIから交通データを取得する参加者向けです．データをR2へ公開する運営者は，[運営者向け導入ガイド](operator-installation.md)を参照してください．

## Webだけを使う場合

Davis Webからダウンロードするだけであれば，アプリのインストールは不要です．運営から案内されたURLをブラウザで開き，参加者共通コードを入力してください．

ログイン後は，schemaの内容を確認しながら検索・絞り込みを行い，必要なファイルを選択してダウンロードできます．

## CLIも使う場合

Davis CLIは，macOS，Windows，Linux用に事前buildされた実行ファイルを使用します．Rust，Cargo，Pythonは必要ありません．installerはOSとCPUを自動判定し，downloadしたファイルのSHA-256 checksumを検証します．

### macOS・Linux

ターミナルで次を実行します．

```bash
curl --proto '=https' --tlsv1.2 -fsSL https://raw.githubusercontent.com/bin-utokyo/davis/main/scripts/install.sh | sh
```

### Windows

PowerShellで次を実行します．

```powershell
irm https://raw.githubusercontent.com/bin-utokyo/davis/main/scripts/install.ps1 | iex
```

installerは旧Python版の`davis-cli`が`uv`で導入されている場合，旧版を削除してから新版へ切り替えます．既にdownloadしたデータ，clone済みrepository，Davisのlogin sessionは削除しません．

## インストールの確認

installerの完了後，新しいターミナルまたはPowerShellを開き，次を実行します．

```text
davis --version
davis --help
```

`davis --help`に`login`，`list`，`info`，`get`が表示されれば導入完了です．

## 参加者ログイン

運営から案内されたDavis WebのURLを指定します．

```text
davis login https://davis-web.davis-bin.workers.dev
```

`Invite code:`と表示されたら，参加者共通コードを入力します．入力内容は画面に表示されません．sessionが有効な間は，共通コードの再入力は不要です．Web browserのsessionとCLIのsessionは別なので，Webでlogin済みでもCLIでは初回loginが必要です．

## データの検索と取得

```text
davis list
davis info routes/Matsuyama
davis get routes/Matsuyama
```

`get`は，commandを実行したdirectory以下へ`data/...`の階層を再現します．Webで必要なファイルを選び，「CLIコマンドをコピー」から取得commandを作ることもできます．

## 更新

CLIは24時間に1回だけ最新版を確認し，新しいreleaseがある場合は通常commandの完了後に案内します．更新確認に失敗しても，データの検索や取得には影響しません．

```text
davis update
```

このcommandを実行すると，現在のversionと最新版を比較し，OSに対応する更新commandを表示します．表示されたinstallerを実行すると最新版へ更新できます．login sessionとdownload済みcacheは維持されます．

## 困ったとき

- `davis`が見つからない場合は，新しいターミナルを開いてください．
- 旧版が呼ばれる場合は，macOS・Linuxでは`which -a davis`，Windowsでは`Get-Command davis -All`で実行fileを確認してください．
- 招待コードが通らない場合は，参加者用コードであることと，案内されたURLが正しいことを確認してください．
- 運営者共通コードを参加者ログインへ入力しないでください．2種類のコードは別管理です．
