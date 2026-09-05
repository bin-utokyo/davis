# Davis Model prototype

この文書は，`davis.analysis/v1alpha1` prototypeの実装範囲と操作方法を示します．

## 現在動作する範囲

* local CSVのencoding，delimiter，列型，欠損，先頭0のinspection
* `AnalysisPlan` (`model.yaml`)と`ComponentManifest`の読込
* component固有JSON Schemaによるconfig検証
* local inputのpath解決，BLAKE3 digest，media type検査
* Pythonまたはnative processの起動とlog保存
* `RunResult`とartifactのpath，size，digest検証
* long形式CSVを使う標準MNLの検証・推定
* CLIとTauri desktop appから同じRust use caseの呼出し
* local component packageの検証，per-user install，一覧，詳細表示，削除
* 決定的component bundle，registry entry，公式registryの生成
* 公式registryからの互換version選択，安全なdownload・検証・install
* 分析project固有componentとinstall済みcomponentの優先探索
* `model`，`transform`，`visualize`のcomponent種別
* Manifestによる出力artifact名，media type，必須条件の検証
* 過去Runのartifactをsize・BLAKE3検証して次のRunへ渡す入力resolver
* CSVへ再現可能な計算列を追加する参考transform component
* 単一・複合key，関係性，未一致方針を明示する複数CSV join
* 先頭0を保護する型推定と明示的schemaによるCSV／Parquet出力
* 1つのmodel.yamlで複数sourceを結合・列選択して推定するtable binding

catalog input，filter，group，任意pipeline DAGは型または拡張点だけを用意しており，まだ実行できません．未実装入力を指定した場合は明示的に失敗します．component作成方法とtable bindingは[`davis-component-authoring.md`](davis-component-authoring.md)に記載します．

## 最小example

```console
davis install component ./components/davis-mnl
davis component list
davis model inspect components/davis-mnl/examples/minimal/choice.csv
davis model validate components/davis-mnl/examples/minimal/model.yaml
davis model plan components/davis-mnl/examples/minimal/model.yaml
davis model run components/davis-mnl/examples/minimal/model.yaml
```

install後はDavis repository外の分析projectでも，同じ`component` IDとversionを指定した`model.yaml`を実行できます．componentは分析projectの`components/`，per-user install領域の順に探索します．application bundle内の組み込みcomponent探索は後続実装です．

```console
davis component inspect davis/mnl --version 0.2.0
davis component remove davis/mnl --version 0.2.0
```

公式registryがreleaseへ公開された後は，`davis install component mnl`または`davis install component davis/mnl --version 0.2.0`で取得できます．registryとbundleの公開契約は[`davis-component-registry.md`](davis-component-registry.md)に記載します．生成・実動作検証・release添付workflowは実装済みで，公式artifactは次のrelease tag公開時に利用可能になります．

per-user install先はmacOSでは`~/Library/Application Support/Davis/components/`，Windowsではlocal application data，Linuxでは`$XDG_DATA_HOME/davis/components/`または`~/.local/share/davis/components/`です．開発・test時は`DAVIS_DATA_HOME`で変更できます．installは`.venv`，`__pycache__`，Git metadata等を除外し，component ID，version，schema，lockfile，symlink，重複を検証してから同一filesystem内でatomicに配置します．

結果は既定で`davis-runs/<run-id>/`へ保存されます．`davis-runs/`は通常のフォルダとして確認でき，repositoryの`.gitignore`によってGitの追跡対象から除外されます．

```text
<run-id>/
├── model.yaml
├── request.json
├── run.json
├── result.json
├── preparation/
│   └── input-0/
│       ├── request.json
│       ├── result.json
│       └── logs/
├── logs/
│   ├── stdout.log
│   └── stderr.log
└── artifacts/
    ├── parameters.csv
    ├── covariance.csv
    ├── metrics.json
    ├── predictions.csv
    ├── prepared/
    │   └── input-0/
    │       ├── transformed.parquet
    │       └── transformation-summary.json
    └── sample-summary.json
```

## Desktop app

desktop appはlocal HTTP serverを実行せず，Tauri IPCから`davis-runtime`を直接呼びます．prebuilt frontendをrepositoryへ含めるため，利用時にNode.jsやpnpmは必要ありません．repository rootから次を実行します．

```console
cargo run -p davis-app
```

React画面自体を変更する開発者だけが，`apps/davis-app`でNode.js packageをinstallしてFrontendを再buildします．

desktop画面は次を提供します．

1. `model.yaml`と`davis-runs`を置くproject workspaceの選択(repository cloneは不要)
2. 複数local CSVの追加と列確認
3. 基準表，join key，関係性，join方式の指定
4. MNLのroleと線形効用termのForm編集
5. 型付きAnalysisPlanから生成した`model.yaml`のpreview，保存，検証，実行
6. 既存`model.yaml`の選択・検証・実行
7. 単一CSVのinspection
8. `ui_schema`が指定するJSON／CSV result view，artifact一覧，run directoryの表示

初版editorは`ui_schema`が`linear-utility` editorを宣言するcomponentと，追加表を基準表へ直接結合するstar型のtable bindingを対象とします．role一覧と必須性はcomponentの`config_schema`，表示名，editor widget，選択肢候補の取得元，入力準備componentは`ui_schema`から取得します．生成YAMLはFrontend固有形式ではなく，Rust側で共通`AnalysisPlan`へdeserializeして契約検証してから保存します．

対応範囲内の既存`model.yaml`は，同じFormへ読み戻して上書きまたは別名保存できます．相対local pathはPlan directoryを基準に解決して表示します．対応範囲外のcomponentや入力表現は内容を失わないYAML modeで開きます．複合keyと追加source間の連鎖joinをFormで編集する機能は後続sliceです．選択肢別termは，`alternative_id` roleへ割り当てたCSV列から上限付きdistinct sampleを取得し，検索可能な候補として表示します．単独のCSV inspection画面は設けず，各入力データカードへ統合します．

推定完了時は結果sectionへ自動scrollします．表示内容はcomponentの`ui_schema.ui:results`がartifact名，title，`key-value`／`table` widgetを指定します．desktopは宣言されたRun artifactだけを安全なpathとsize上限の下で読み，汎用rendererで表示します．表示宣言がないartifactも一覧と結果folderから参照できます．

## Component package

componentは最低限，次を含みます．

```text
component/
├── component-manifest.yaml
├── pyproject.tomlまたはnative executable
├── lockfile
├── schemas/
│   ├── config.schema.json
│   └── ui.schema.json
└── src/
```

RunnerはManifestの`runtime.command`へ`request_argument`と`request.json`の絶対pathを追加して起動します．processは指定されたoutput directoryへ`run-result.json`を書きます．artifact pathはoutput directoryからの安全な相対pathでなければなりません．

localまたはregistry componentは任意codeを実行するため，現在のprototypeでは信頼できるcomponentだけをinstallしてください．sandboxとregistry署名は後続実装です．現在のPython componentは実行時に`uv`を必要とし，Davis管理runtimeの自動installはまだ実装していません．

ComponentManifestの`requires_davis`はcomponentが必要とするDavis contractのSemVer条件です．本体のrelease versionとは独立しており，互換性が維持されている限り，本体のminor updateに合わせて機械的に上げません．MNL 0.2.0は`>=0.3.5`を宣言します．旧packageの`model-manifest.yaml`と`davis.model/v1alpha1`も後方互換として読み込めます．
