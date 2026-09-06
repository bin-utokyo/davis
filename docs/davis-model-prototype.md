# Davis Model prototype

この文書は，`davis.analysis/v1alpha1` prototypeの実装範囲と操作方法を示します．

## 現在動作する範囲

* local CSVのencoding，delimiter，列型，欠損，先頭0のinspection
* `AnalysisPlan` (`model.yaml`)と`ComponentManifest`の読込
* component固有JSON Schemaによるconfig検証
* local inputのpath解決，BLAKE3 digest，media type検査
* Davis Catalog inputの検索，1 click download，共有cacheからのpath解決・整合性検査
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

Catalog inputは，先に`davis login <URL>`を済ませるとDesktopから検索・downloadでき，`kind: catalog`の論理参照としてPlanへ保存されます．CLIとDesktopはOSごとのDavis data directoryにある同じcontent-addressed cacheを使うため，作業directoryを変えても再利用できます．revision pin，filter，group，任意pipeline DAGはまだ実行できません．component作成方法とtable bindingは[`davis-component-authoring.md`](davis-component-authoring.md)に記載します．

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
davis component inspect davis/mnl --version 0.3.1
davis component remove davis/mnl --version 0.3.1
```

公式registryがreleaseへ公開された後は，`davis install component mnl`または`davis install component davis/mnl --version 0.3.1`で取得できます．registryとbundleの公開契約は[`davis-component-registry.md`](davis-component-registry.md)に記載します．生成・実動作検証・release添付workflowは実装済みで，公式artifactは次のrelease tag公開時に利用可能になります．

## `Tohoku_History`実データexample

`components/davis-mnl/examples/tohoku-history/model.yaml`は，Davis Catalogの`df_ex_var.csv`と`df_individual.csv`を直接参照します．GUIで開く場合は，2つの入力をCatalogから選び，`individual_id`と`time`を複合keyとして結合します．ケースキーも同じ2列，選択肢IDは`city`，実際に選ばれた選択肢IDは`target`です．

```console
davis login <Davis Web URL>
davis get Tohoku_History --file df_ex_var.csv --file df_individual.csv
davis model run components/davis-mnl/examples/tohoku-history/model.yaml
```

このexampleは動作確認時間を抑えるため，`development_case_limit: 200`で入力順の先頭200ケースだけを推定します．これは標本抽出法ではなく開発用上限です．研究上の本推定ではこの設定を削除し，分析目的に沿った標本作成を明示的なtransformとして記録してください．実データそのものはGitへ追加しません．

per-user install先はmacOSでは`~/Library/Application Support/Davis/components/`，Windowsではlocal application data，Linuxでは`$XDG_DATA_HOME/davis/components/`または`~/.local/share/davis/components/`です．開発・test時は`DAVIS_DATA_HOME`で変更できます．installは`.venv`，`__pycache__`，Git metadata等を除外し，component ID，version，schema，lockfile，symlink，重複を検証してから同一filesystem内でatomicに配置します．

結果は既定で`davis-runs/<run-id>/`へ保存されます．`run-id`は`朝ピーク-nl__20260906-153012__a1b2c3d4`のように，意味のあるprefix，実行日時，一意suffixから構成されます．prefixはAnalysis Planの`run.label`，未指定なら`name`から生成されます．`run.label`はGUIの`Run name (folder prefix)`，またはYAMLから個々に設定できます．pathとして危険な記号は自動的に`-`へ正規化され，日時とsuffixは衝突回避と追跡のためDavisが付与します．`davis-runs/`は通常のフォルダとして確認でき，repositoryの`.gitignore`によってGitの追跡対象から除外されます．

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

React画面を変更した場合は，`pnpm run build`の完了後にDesktopを再起動してください．`cargo run -p davis-app`はrepositoryに含まれるprebuilt frontendを使うため，起動したままでは新しい画面へ切り替わりません．

desktop画面は次を提供します．

1. `model.yaml`と`davis-runs`を置くproject workspaceの選択(repository cloneは不要)
2. 複数local CSVの追加と列確認
3. 基準表，join key，関係性，join方式の指定
4. `davis.ui/v1`による任意componentの入力・設定section編集
5. 型付きAnalysisPlanから生成した`model.yaml`のpreview，保存，検証，実行
6. 既存`model.yaml`の選択・検証・実行
7. 単一CSVのinspection
8. `presentation.ui`が指定するJSON／CSV result view，artifact一覧，run directoryの表示

Desktopは`presentation.ui.version: davis.ui/v1`を1つの共通UI契約として扱います．入力slotと設定sectionはManifestに並べたwidgetから構成し，モデルIDや画面種別によるFrontend分岐は持ちません．各入力slotでは共通`table-binding` UIにより，追加表を基準表へ直接結合するstar型bindingを作成できます．設定項目と必須性はcomponentの`configuration.schema`，表示名，widget，参照元，入力準備componentは`presentation.ui`から取得します．生成YAMLはFrontend固有形式ではなく，Rust側で共通`AnalysisPlan`へdeserializeして契約検証してから保存します．

対応範囲内の既存`model.yaml`は，同じFormへ読み戻して上書きまたは別名保存できます．相対local pathはPlan directoryを基準に解決して表示します．対応範囲外のcomponentや入力表現は内容を失わないYAML modeで開きます．複合keyと追加source間の連鎖joinをFormで編集する機能は後続sliceです．選択肢別termは，`alternative_id` roleへ割り当てたCSV列から上限付きdistinct sampleを取得し，検索可能な候補として表示します．単独のCSV inspection画面は設けず，各入力データカードへ統合します．

推定完了時は結果sectionへ自動scrollします．表示内容はcomponentの`presentation.ui.results`がartifact名，title，`key-value`／`table` widgetを指定します．desktopは宣言されたRun artifactだけを安全なpathとsize上限の下で読み，汎用rendererで表示します．表示宣言がないartifactも一覧と結果folderから参照できます．

## Component package

componentは最低限，次を含みます．

```text
component/
├── component.yaml
└── 実program
```

RunnerはManifestの`runtime.command`へ`request_argument`と`request.json`の絶対pathを追加して起動します．processは指定されたoutput directoryへ`run-result.json`を書きます．artifact pathはoutput directoryからの安全な相対pathでなければなりません．

localまたはregistry componentは任意codeを実行するため，現在のprototypeでは信頼できるcomponentだけをinstallしてください．sandboxとregistry署名は後続実装です．Runtimeの正規形は言語非依存の`executor: process`です．現在のPython componentは実行時に`uv`を必要とし，Davisは一般言語環境を自動installせず，Manifestの`requirements`に基づいて不足commandと導入方法を案内します．

ComponentManifestの`requires_davis`はcomponentが必要とするDavis contractのSemVer条件です．本体のrelease versionとは独立しており，互換性が維持されている限り，本体のminor updateに合わせて機械的に上げません．MNL 0.3.1は`>=0.5.0`を宣言します．正規の`component.yaml`はconfig schemaとpresentationをinlineに保持でき，旧packageの`component-manifest.yaml`，`model-manifest.yaml`，`davis.component/v1alpha1`，`davis.model/v1alpha1`も後方互換として読み込めます．
