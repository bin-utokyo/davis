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
* local component packageの検証，per-user install，一覧，詳細表示，削除
* 分析project固有componentとinstall済みcomponentの優先探索

catalog input，run artifact input，join，filter，Parquetへの内部materializeは型または拡張点だけを用意しており，まだ実行できません．未実装入力を指定した場合は明示的に失敗します．

## 最小example

```console
davis install component ./components/davis-mnl
davis component list
davis model inspect components/davis-mnl/examples/minimal/choice.csv
davis model validate components/davis-mnl/examples/minimal/model.yaml
davis model plan components/davis-mnl/examples/minimal/model.yaml
davis model run components/davis-mnl/examples/minimal/model.yaml
```

install後はDavis repository外の分析projectでも，同じ`component` IDとversionを指定した`model.yaml`を実行できます．componentは分析projectの`components/`，per-user install領域の順に探索します．公式registryとapplication bundle内の組み込みcomponent探索は後続実装です．

```console
davis component inspect davis/mnl --version 0.1.0
davis component remove davis/mnl --version 0.1.0
```

per-user install先はmacOSでは`~/Library/Application Support/Davis/components/`，Windowsではlocal application data，Linuxでは`$XDG_DATA_HOME/davis/components/`または`~/.local/share/davis/components/`です．開発・test時は`DAVIS_DATA_HOME`で変更できます．installは`.venv`，`__pycache__`，Git metadata等を除外し，component ID，version，schema，lockfile，symlink，重複を検証してから同一filesystem内でatomicに配置します．

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

local componentは任意codeを実行するため，現在のprototypeでは信頼できるcomponentだけをinstallしてください．sandbox，署名，公式registryは後続実装です．現在のPython componentは実行時に`uv`を必要とし，Davis管理runtimeの自動installはまだ実装していません．
