# Davis Site Admin向け運用ガイド

[English](site-admin-installation_en.md)

このガイドは，Davis Web deployment全体のaccess groupとdataset download権限を管理するSite Admin向けです．実データやschemaを日常的に更新する方は，[運営者向け導入ガイド](operator-installation.md)を参照してください．データを取得するだけの場合は，[参加者向け導入ガイド](participant-installation.md)を参照してください．

## 1. 役割の違い

| 役割 | 認証情報 | 主な操作 |
| --- | --- | --- |
| 参加者 | access groupの参加者コード | 公開catalogの検索，許可されたdatasetのdownload |
| 運営者 | 同じaccess groupの運営コード | Object upload，担当datasetの初期割当，catalog公開 |
| Site Admin | deployment共通のSite Adminコード | access group作成，dataset権限変更，既存storage移行 |
| Cloudflare管理者 | Cloudflare Accountの認証 | Worker・Secret・R2 bindingの設定とdeploy |

Site Adminは権限変更とstorage移行について通常の運営者より強い権限を持ちますが，参加者としてdownloadする権限や，Cloudflare AccountへloginしてWorker Secretを変更する権限を兼ねません．少人数に限定し，日常の`push`や`publish`には運営コードを使用してください．

## 2. 初期設定

Davis CLI v0.5.9以降を使用します．

```bash
davis --version
davis admin --help
```

deployment側では，推測困難なSite AdminコードをWorker Secretとして登録します．コードをrepository，通常の環境変数，issue，Pull Requestへ保存しないでください．

```bash
cd web/davis-web
pnpm exec wrangler secret put DAVIS_ADMIN_CODE
pnpm deploy
```

`DAVIS_ADMIN_ACCESS_REVISION`はSite Admin sessionの一括失効に使用します．コード流出時や担当者交代時は，`DAVIS_ADMIN_CODE`を交換し，revisionも新しい値へ変更して再deployしてください．参加者・運営者sessionは，それぞれ別のrevisionで管理されるため影響を受けません．

旧参加者コードと旧運営コードを1つのgroupとして継続する場合は，`DAVIS_LEGACY_GROUP_ID`を設定します．移行時に参加者・運営者のaccess revisionも更新すると，旧sessionを残さず再loginさせられます．

## 3. Site Admin login

Site Adminコードは対話promptへ入力します．command引数やshell履歴へ記載しないでください．

```bash
davis admin login <Davis Web URL>
davis admin status
```

CLIが保存するのは期限付きSite Admin sessionです．Site Adminコード自体はCLIに保存しません．作業終了時や共用端末ではsessionを削除してください．

```bash
davis admin logout
```

## 4. access groupを作成する

1つのaccess groupには，参加者コードと運営コードが1対1で対応します．group IDは小文字英数字とhyphenを使用し，年度や組織を識別できる安定した名前にします．

```bash
davis admin group-create municipality-a-2026
```

2つのコードは作成時に一度だけ表示されます．再表示できないため，参加者コードと運営コードを別々の安全な保管先へ記録してください．運営コードを参加者へ配布しないでください．

groupの運営コードで新しい未公開datasetを最初に`push`すると，そのdatasetは同じgroupへ自動的に割り当てられます．すでに別groupへ割り当てられたdatasetを通常の運営者が奪うことはできません．

```bash
davis operator login <Davis Web URL>
davis push <dataset ID>
```

## 5. datasetのdownload権限を変更する

Site Adminは，1つのdatasetへ複数groupを許可できます．`dataset-access`は追加操作ではなく，指定したgroup集合への置換です．既存groupを残す場合は，そのgroupもすべて列挙してください．

```bash
davis admin dataset-access network/matsuyama \
  --group municipality-a-2026 \
  --group research-team-2026
```

変更後は，新しいDownload Grantだけでなく，発行済みGrantを使用するdownload時にも現在の権限を再確認します．権限を外されたgroupは，期限内の参加者sessionを持っていても対象datasetを取得できません．すでに端末へ保存されたcopyは技術的に回収できません．

現行CLIは，少なくとも1つのgroupを要求します．datasetを全groupから一時的に非公開にする操作や，group自体の削除・コード再発行は提供していません．参加者コードの漏えい時は新しいgroupを作成し，旧groupを含まない許可group集合へ各datasetを変更してください．旧groupは登録されたままですが，許可datasetがなければ実データを取得できません．運営コードが漏えいした場合は，旧groupが新規datasetを割り当ててcatalogを公開できるため，公開作業を停止してdeployment管理者へ連絡してください．現行CLIだけでは完全失効できません．private R2 Objectの`access/control.json`を通常の運営者が手作業で編集してはいけません．

## 6. R2 Objectのgzip保管

v0.5.9以降のCLIは，新規・変更Objectを運営者端末でgzip圧縮してからmultipart uploadします．Workerは圧縮済みObjectをR2へ保存し，download時には`Content-Encoding: gzip`を付けてそのまま配信します．ブラウザまたはCLIが端末側で自動解凍するため，利用者が保存するfile名と内容は従来どおりです．

圧縮済みObjectでは，元fileのbyte位置とgzip上のbyte位置が一致しないためRange downloadを提供しません．常にfull responseとして取得します．未圧縮の旧Objectは従来どおりRange downloadできます．

既存Objectを移行する場合は，対象catalogの全Objectがlocal content-addressed storeに存在する端末から実行します．標準の場所はrepository内の`.davis/cache`です．各local ObjectをBLAKE3とsizeで確認し，端末でgzip化してR2へuploadした後，圧縮representationを確定してraw copyを削除します．

```bash
git status
davis admin storage-compress --yes
```

別のstoreを使用する場合は明示します．

```bash
davis admin storage-compress --yes --store /absolute/path/to/cache
```

この操作は同じObjectに対して再実行できます．圧縮済みObjectは再uploadせず，途中で失敗したObjectのraw copyは残ります．ただし，全catalog Objectを読み取るため時間とlocal CPUを使用します．移行前にlocal cacheの完全性と十分な一時disk容量を確認してください．

## 7. 既存コードをlegacy groupへ移す

既存の`DAVIS_INVITE_CODE`と`DAVIS_OPERATOR_CODE`を継続する場合は，たとえば`bmss26`をlegacy groupに設定します．

```text
DAVIS_LEGACY_GROUP_ID=legacy-group-id
```

移行手順は次のとおりです．

1. 短い保守時間を設定し，現在公開中のdataset IDを控える
2. `DAVIS_LEGACY_GROUP_ID`を設定する
3. `DAVIS_ACCESS_REVISION`と`DAVIS_OPERATOR_ACCESS_REVISION`を更新してWorkerをdeployする
4. Site Adminでloginし，各公開datasetへ`davis admin dataset-access <dataset ID> --group <legacy group ID>`を実行する
5. 旧参加者コードと旧運営コードで再loginし，download・運営操作を確認する
6. 新しいgroupを作成し，必要なdatasetを段階的に割り当てる

この移行は，既存コードを無効にせずaccess groupへ位置付けるためのものです．revision更新により旧sessionは失効しますが，同じコードで再loginできます．

## 8. credential管理と事故対応

- Site Adminコード，参加者コード，運営コードをrepositoryへcommitしないでください．
- Site Adminコードと運営コードは，参加者向け資料や一般の連絡先へ記載しないでください．
- `group-create`で表示されたコードは再取得できないため，組織のpassword manager等へ直ちに保存してください．
- Site Admin session流出時は`DAVIS_ADMIN_ACCESS_REVISION`を更新してください．
- Site Adminコード流出時はSecretとrevisionの両方を交換してください．
- 参加者groupコード流出時は，新groupを作成し，旧groupを含まない許可group集合へ全対象datasetを変更してください．
- 運営groupコード流出時は公開作業を停止し，deployment管理者へ連絡してください．現行CLIにはgroup削除・コード再発行がないため，grant移行だけでは完全失効になりません．
- R2 credentialをSite Adminや通常の運営端末へ配布する必要はありません．

## 9. 日常運用の確認表

```bash
davis admin status
davis operator status
git status
```

Site Adminは権限変更の目的と対象dataset・groupを記録し，可能なら別担当者の確認を受けてください．通常のデータ更新，review，`push`，`publish`はSite Admin sessionではなく運営sessionで行います．
