# Davis Model prototype

この文書は，`davis.analysis/v1alpha1` prototypeの実装範囲と操作方法を示します．

## 現在動作する範囲

* local CSVのencoding，delimiter，列型，欠損，先頭0のinspection
* `AnalysisPlan` (`model.yaml`)と`ModelManifest`の読込
* component固有JSON Schemaによるconfig検証
* local inputのpath解決，BLAKE3 digest，media type検査
* Pythonまたはnative processの起動とlog保存
* `RunResult`とartifactのpath，size，digest検証
* long形式CSVを使う標準MNLの検証・推定
* CLIとTauri desktop appから同じRust use caseの呼出し

catalog input，run artifact input，join，filter，Parquetへの内部materializeは型または拡張点だけを用意しており，まだ実行できません．未実装入力を指定した場合は明示的に失敗します．

## 最小example

```console
davis model inspect components/davis-mnl/examples/minimal/choice.csv
davis model validate components/davis-mnl/examples/minimal/model.yaml
davis model plan components/davis-mnl/examples/minimal/model.yaml
davis model run components/davis-mnl/examples/minimal/model.yaml
```

結果は既定で`davis-runs/<run-id>/`へ保存されます．`davis-runs/`は通常のフォルダとして確認でき，repositoryの`.gitignore`によってGitの追跡対象から除外されます．

```text
<run-id>/
├── model.yaml
├── request.json
├── run.json
├── result.json
├── logs/
│   ├── stdout.log
│   └── stderr.log
└── artifacts/
    ├── parameters.csv
    ├── covariance.csv
    ├── metrics.json
    ├── predictions.csv
    └── sample-summary.json
```

## Desktop app

desktop appはlocal HTTP serverを実行せず，Tauri IPCから`davis-runtime`を直接呼びます．prebuilt frontendをrepositoryへ含めるため，利用時にNode.jsやpnpmは必要ありません．repository rootから次を実行します．

```console
cargo run -p davis-app
```

React画面自体を変更する開発者だけが，`apps/davis-app`でNode.js packageをinstallしてFrontendを再buildします．

初期画面は次を提供します．

1. Davis repositoryの選択
2. local CSVのinspection
3. 既存`model.yaml`の選択・検証
4. model componentの実行
5. run directoryとartifact一覧の表示

GUI上で効用termを編集して`model.yaml`を生成する機能は次のsliceです．

## Component package

componentは最低限，次を含みます．

```text
component/
├── model-manifest.yaml
├── pyproject.tomlまたはnative executable
├── lockfile
├── schemas/
│   ├── config.schema.json
│   └── ui.schema.json
└── src/
```

RunnerはManifestの`runtime.command`へ`request_argument`と`request.json`の絶対pathを追加して起動します．processは指定されたoutput directoryへ`run-result.json`を書きます．artifact pathはoutput directoryからの安全な相対pathでなければなりません．

local componentは任意codeを実行するため，現在のprototypeでは信頼できるcomponentだけを登録してください．sandboxと署名は後続実装です．
