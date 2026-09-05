# Davis software registry

## 1. 利用者向けの導入

DavisはCLIを最初に導入する小さなbootstrapとします．Davis Desktopを使うためにGitHub repositoryをcloneする必要はありません．任意のworking directoryから次を実行します．

```console
davis install desktop
davis desktop
```

既存の仕様書にある`davis install app`もaliasとして利用できます．導入済みのDesktopとcomponentは，次のcommandでまとめて確認できます．

```console
davis installed
davis installed --json
```

DesktopはOS標準のDavis user data directory以下の`software/desktop/<version>`へ保存されます．`DAVIS_DATA_HOME`を指定した場合は，そのdirectoryを起点にします．このため，install時または起動時のcurrent working directoryには依存しません．

## 2. Registry契約

公式registryはGitHub Releaseの`software-registry.json`です．schema version 1は次の形です．

```json
{
  "schema_version": 1,
  "packages": [
    {
      "id": "desktop",
      "name": "Davis Desktop",
      "version": "0.5.0",
      "requires_davis": ">=0.5.0",
      "artifacts": [
        {
          "target": "aarch64-apple-darwin",
          "url": "davis-desktop-macos-aarch64.tar.gz",
          "size": 123456,
          "blake3": "blake3:...",
          "entrypoint": "davis-app"
        }
      ]
    }
  ]
}
```

`requires_davis`はpackageが要求するCLI・software registry契約のSemVer条件です．この契約を導入した最初のDesktop packageは`>=0.5.0`を要求します．componentの`requires_davis`とは同じ記法ですが，各packageは独立して互換性を宣言します．

CLIは自身のversionと実行環境に合う最新packageを選択し，download sizeとBLAKE3 digestを検証してから安全に展開します．絶対path，親directoryへの脱出，symlink，未定義entrypointを拒否し，検証がすべて成功した場合だけinstall記録を確定します．Registry URLはHTTPSのみを許可します(local test用のlocalhostを除きます)．

## 3. Release生成

tagを付けたrelease workflowはCLIとは別に各OS・CPU向けのDesktop executableをbuildし，次を同じGitHub Releaseへ公開します．

- `davis-desktop-<os>-<arch>.tar.gz`
- `software-registry.json`
- 既存のCLI，component bundle，`component-registry.json`，checksum

DesktopとCLIは同じrelease versionに揃えます．DesktopはCLIに埋め込まず，必要な人だけが取得します．将来Python等のmanaged runtimeを追加するときも，`packages`に別IDとplatform artifactを追加し，同じ検証・保存境界を再利用します．
