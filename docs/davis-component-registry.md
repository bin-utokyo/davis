# Davis Component Registry

この文書は，人間とAIが公式model componentを公開・installするためのregistry契約を示します．現在はregistry clientまで実装済みで，公式releaseへのregistryとbundle添付は未実施です．

## 利用者command

```console
davis install component mnl
davis install component davis/mnl --version 0.1.0
```

CLIは既存directoryを指定された場合はlocal packageとしてinstallし，それ以外は公式名またはcomponent IDとしてregistryを検索します．registry URLは次の順で決定します．

1. `--registry <URL>`
2. `DAVIS_COMPONENT_REGISTRY_URL`
3. Davis GitHub Releasesの`component-registry.json`

開発用のlocalhostを除き，HTTPは拒否してHTTPSだけを利用します．

## Registry contract

```json
{
  "schema_version": 1,
  "components": [
    {
      "name": "mnl",
      "id": "davis/mnl",
      "version": "0.1.0",
      "requires_davis": ">=0.3.0, <0.4.0",
      "bundle": {
        "url": "davis-mnl-0.1.0.tar.gz",
        "size": 12345,
        "blake3": "blake3:..."
      }
    }
  ]
}
```

`name`は短いinstall名，`id`はModelManifestの完全なIDです．同じ`name`と`id`で複数versionを掲載できます．versionを省略した場合，実行中のDavis versionが`requires_davis`を満たす最新SemVerを選択します．相対bundle URLはregistry URLを基準に解決します．

## Bundle contract

bundleはgzip圧縮tarで，archive rootに`model-manifest.yaml`を置きます．

```text
model-manifest.yaml
pyproject.toml
uv.lock
schemas/
src/
```

親directory，絶対path，symlink，hard link，device等を含められません．CLIは次を確認してからper-user component storeへatomicにinstallします．

1. registryは2 MiB以下です．
2. 圧縮bundleは512 MiB以下で，registryのsizeと一致します．
3. bundle全体のBLAKE3がregistryと一致します．
4. 展開後の通常file合計は2 GiB以下です．
5. archive pathが展開先から脱出しません．
6. ModelManifestのIDとversionがregistry entryと一致します．
7. config schema，UI schema，lockfile等のpackage参照が安全で実在します．

registryのdigestはtransport破損とrelease metadataの不一致を検出しますが，registry自体の署名ではありません．公式releaseの署名とtrust policyは後続実装です．local componentと同様，信頼できないcomponentはinstall・実行しないでください．

## 公開者の責務

component公開者は，runtimeに必要なfileだけをbundleへ含めます．`.venv`，`__pycache__`，Git metadata，build出力を含めません．release workflowはbundleを作成した後にsizeとBLAKE3を計算し，registry entryとbundleを同じGitHub Releaseへ添付します．

Python runtime，native executable等のplatform差が必要になった場合は，registry schemaの次versionでtarget情報を追加します．`schema_version: 1`ではplatform非依存packageだけを対象とします．
