# Davis Component Authoring Guide

この文書を読むと，自分で作った計算programをDavisの画面やCLIから実行できるようになります．統計モデルだけでなく，「2つのCSVを結合する」「説明変数を計算する」「推定結果を図にする」といった処理も追加できます．Python以外の言語でも構いません．

Davisでは，このような追加処理一式を**component**と呼びます．componentは，普通のprogramに「何を入力し，どの設定を使い，何を出力するか」という説明書を付けたものです．その説明書が`component.yaml`です．

初学者は「最初の作成手順」と「それぞれの言葉の意味」から読んでください．既に実装したい処理が決まっている方やAIは，「Manifest」以降を仕様書として使えます．

## それぞれの言葉の意味

| 言葉 | 意味 | 例 |
| --- | --- | --- |
| component | Davisから実行できるようにまとめた計算処理一式 | MNL推定，CSV結合 |
| Manifest | componentの説明書．filenameは`component.yaml` | 入力はCSV，出力はparameters.csv |
| input | programへ渡す入力file | `persons.csv` |
| config | 実行する人が毎回決める設定 | ID列名，説明変数，反復回数 |
| artifact | programが作った出力file | 推定係数CSV，指標JSON |
| runtime command | programを起動するcommand | `python -m my_model` |
| Analysis plan | 1回の実行内容を保存するYAML | 使用file，列，設定 |
| schema | configに何を書けるかを示す規則 | `max_iterations`は1以上の整数 |
| presentation | configや結果をGUIでどう見せるかという補助情報 | 結果CSVをtable表示 |

関係は次のとおりです．

```text
component.yaml  ──「このprogramは何を受け取り，何を返すか」
実program       ── 実際の計算
analysis.yaml   ──「今回はどのfileと設定で実行するか」
       │
       └── Davisが実行し，結果と実行記録をdavis-runsへ保存
```

## 基本構造

最小componentは，説明書と実programの2つで構成できます．次の`model.py`という名前は一例であり，R，Julia，Node.js，native executable等でも構いません．

```text
component/
├── component.yaml
└── model.py
```

Davisは入力fileの場所を解決し，programを起動し，logと結果を保存します．さらに，宣言した出力が本当に作られたかを確認します．モデル固有の計算は実programが担当します．

## 最初の作成手順

まず，componentを入れる新しいdirectoryと`component.yaml`を自動生成します．次の例は，Python module `my_component`を起動するデータ加工componentです．既に同名のdirectoryがある場合は上書きしません．

```console
davis component scaffold ./my-component \
  --id example/my-component \
  --kind transform \
  --command python \
  --command -m \
  --command my_component
```

次に，生成されたdirectoryへ実programを追加します．`component.yaml`を書き換えたら，installする前に説明書の間違いを検査します．

```console
davis component validate ./my-component
davis install component ./my-component
davis component inspect example/my-component
davis model run ./analysis.yaml
```

`validate`が確認するのは，YAMLの形式，IDとversion，schema，参照file等です．計算内容が正しいかまでは判断しないため，小さな入力例を使ったtestも作ってください．実行時には，Python等の必要なcommandがPCにあるかも確認されます．Davisの開発repositoryへcomponentを置く必要はありません．

### AIに作成を頼む場合

AIにはこの文書全体と，少なくとも次の情報を渡してください．Davisの内部実装や開発経緯を説明する必要はありません．

1. componentの目的と`kind`
2. 入力slotごとの名前，media type，意味
3. 設定項目と制約
4. 出力artifactごとの名前，media type，意味
5. 利用可能なruntime commandと必要な外部環境

依頼文では「推測でDavis独自fieldを追加せず，このガイドに書かれた契約だけを使う」「完成後に`davis component validate`と既知入力による実行手順を示す」と指定してください．AIが生成した任意codeは，実行前に内容を確認してください．

次の依頼文を出発点にできます．

```text
添付したDavis Component Authoring GuideだけをDavis固有仕様の根拠として，
次の処理を行うcomponentを作成してください．

目的: (行いたい推定や計算)
入力: (file名，形式，各列の意味)
実行時に変更したい設定: (説明変数等)
出力: (必要な表や指標)
使用できる言語・command: (Python，R等)

component.yaml，実program，最小sample，Analysis plan，testを作成してください．
未記載のDavis独自fieldは推測で追加しないでください．
davis component validateとsample実行の手順も示してください．
```

## Manifest

Manifestは「このcomponentの取扱説明書」です．利用者が毎回選ぶ値はManifestへ直接書かず，`configuration.schema`で項目だけ定義します．実際に選んだ値は後述するAnalysis planへ保存します．

```yaml
api_version: davis.component/v1
id: example/accessibility
name: Accessibility calculator
version: 0.1.0
kind: transform
requires_davis: ">=0.5.0"

runtime:
  executor: process
  command: ["uv", "run", "--frozen", "python", "-m", "accessibility"]
  request_argument: "--request"
  lockfile: uv.lock
  requirements:
    - command: uv
      version: ">=0.8"
      install:
        macos: https://docs.astral.sh/uv/getting-started/installation/
        windows: https://docs.astral.sh/uv/getting-started/installation/
        linux: https://docs.astral.sh/uv/getting-started/installation/

operations: [transform]
inputs:
  - name: persons
    media_types: [text/csv]
    required: true

configuration:
  schema:
    type: object
    required: [person_id]
    properties:
      person_id:
        type: string

presentation:
  ui:
    ui:editor: generic
    ui:results:
      - artifact: explanatory_variables
        title: Explanatory variables
        widget: table

outputs:
  artifacts:
    explanatory_variables:
      media_types: [text/csv]
      required: true
```

`kind`は`model`，`transform`，`visualize`のいずれかです．省略時は既存Manifestとの互換性のため`model`です．`operations`の名前はcomponentが定義します．`requires_davis`はDavis本体のrelease versionから自動生成せず，利用するcontractの互換範囲をcomponent作者が宣言します．

`runtime.executor: process`はprogramming言語に依存しません．`command`が`request.json`を読み，`run-result.json`を返せば，Python，R，Julia，Node.js，Java，Rust，C++等を同じcomponentとして実行できます．Davisは一般言語環境をinstallしません．`requirements`に必要commandと任意のSemVer条件，OS別の導入案内を宣言し，Davisは実行前に存在とversionを確認します．`version_arguments`を省略すると`--version`を使用します．

旧Manifestの`runtime.kind: python`と`runtime.kind: native`は，どちらもprocess実行として後方互換で読み込まれます．

desktop Form editorを提供するcomponentは，`presentation.ui`で`ui:editor`を宣言します．標準MNLの`linear-utility` editorは，roleの表示名を`roles.ui:labels`，選択肢候補を得るroleを`terms.ui:alternativesFromRole`，table bindingに使うtransformを`ui:inputPreparation`から読みます．設定項目と必須性の正本は引き続き`configuration.schema`です．presentationは契約の妥当性を緩めず，表示方法だけを補います．

大きなschemaを分割したい場合は，inline値の代わりに安全なpackage相対pathを指定できます．JSONとYAMLの両方を利用できます．inlineと参照を同時に指定することはできません．

```yaml
configuration:
  schema_ref: schemas/config.schema.json
presentation:
  ui_ref: schemas/ui.schema.json
```

旧Manifestのtop-level `config_schema`と`ui_schema`も同じ外部参照として解決されます．新規componentでは，Web上のLLM，人間，GUIのいずれからも1ファイルを扱いやすいinline形式を標準とします．

結果をdesktop内に表示する場合は，`ui:results`へartifact名，title，widgetを列挙します．`key-value`はJSON object，`table`はCSVを表示します．これは成果物の中央契約を増やすものではなく，Manifestで宣言済みのartifactをどう提示するかというcomponent固有のhintです．未対応widgetや大きすぎるartifactは無理に表示せず，artifact一覧へ残します．

通常は`inputs`でslot名を固定します．CSV joinのように利用者が任意名の追加inputを与えるcomponentだけは，`additional_inputs.media_types`で許可する形式を限定できます．追加inputもDavisがpath，media type，size，BLAKE3を解決・記録してからcomponentへ渡します．

`outputs.artifacts`を宣言したcomponentは，未宣言artifact，必須artifactの欠落，異なるmedia typeを返せません．Davisは各成果物のpath，size，BLAKE3も検証して`result.json`へ記録します．既存componentの`outputs.standard`と`outputs.extensions`は引き続き読めます．

## Analysis plan

Analysis planは「今回の実験条件」です．同じcomponentでも，入力fileや説明変数を変えるたびに別のplanとして保存できます．このため，`component.yaml`を毎回編集して実験履歴の代わりにする必要はありません．

```yaml
api_version: davis.analysis/v1alpha1
name: calculate-accessibility
component:
  id: example/accessibility
  version: 0.1.0
  operation: transform
inputs:
  persons:
    kind: local
    path: persons.csv
config:
  person_id: person_id
```

既存planの`model.component`形式も互換です．新しい汎用表記ではtop-levelに`component`，その中に`id`を書けます．どちらも同じAnalysisPlanとして実行されます．

## Process contract

ここからは実programを書く人とAIが守る規則です．GUIだけで既存componentを使う人は読み飛ばせます．Davisとprogramは，関数呼出しではなく2つのJSON fileで情報を交換します．そのため，programming言語を限定しません．

DavisはManifestの`runtime.command`をcomponent directoryで起動し，`request_argument`の直後に`request.json`の絶対pathを渡します．componentは次を行います．

1. `request.json`を読みます．
2. `inputs.<name>.resolved.path`だけから入力を読みます．
3. `output_directory`以下へ成果物を書きます．
4. `output_directory/run-result.json`を書きます．
5. 成功時は終了code 0，失敗時は非0を返します．

例えば，programには次のようなJSONが渡ります．`source`は利用者が指定した元情報，`resolved`はDavisが検証した実fileです．programは必ず`resolved.path`を読んでください．

```json
{
  "api_version": "davis.run/v1alpha1",
  "run_id": "run_123456",
  "operation": "transform",
  "component": {
    "id": "example/accessibility",
    "version": "0.1.0",
    "kind": "transform",
    "manifest_path": "/absolute/path/component.yaml",
    "source_digest": "blake3:..."
  },
  "inputs": {
    "persons": {
      "source": {"kind": "local", "path": "persons.csv", "read": null},
      "resolved": {
        "path": "/absolute/path/persons.csv",
        "object_id": "blake3:...",
        "size": 1234,
        "media_type": "text/csv"
      }
    }
  },
  "config": {"person_id": "person_id"},
  "output_directory": "/absolute/path/davis-runs/run_123456/artifacts"
}
```

実programは，例えば`python -m accessibility --request /.../request.json`のように起動されます．`--request`の値からJSONを読み，`run_id`と`output_directory`を取り出します．入力fileや出力directoryをcomponent directoryからの相対pathだと仮定しないでください．

成功結果の最小例です．

```json
{
  "api_version": "davis.result/v1alpha1",
  "run_id": "requestと同じrun ID",
  "status": "succeeded",
  "artifacts": {
    "explanatory_variables": {
      "path": "explanatory-variables.csv",
      "media_type": "text/csv"
    }
  },
  "extensions": {}
}
```

componentがsizeやdigestを計算する必要はありません．Davisが実ファイルから計算して記録します．artifact pathは`output_directory`からの安全な相対pathに限定されます．

## Runを接続する

前処理の出力は，絶対pathではなく`run_artifact`で次の処理へ渡せます．

```yaml
inputs:
  choice_data:
    kind: run_artifact
    run_id: run_123456
    artifact: transformed_table
```

Davisは同じrun rootから`result.json`を読み，artifact名，成功状態，media type，size，BLAKE3，安全なpathを再検証します．記録後にファイルが変更されていた場合は次の処理を開始しません．

現段階では1つ目のRunを実行し，得られたrun IDを2つ目のplanへ記入します．複数処理を1ファイルでDAGとして実行するpipeline記法はまだありません．

## 推定時の複数source binding

モデル入力を作る目的では，transformを先に手動実行する必要はありません．`table_binding`をモデルのinput slotへ指定すると，Davisが推定直前に複数sourceを結合し，選択列だけをParquetへmaterializeしてモデルcomponentへ渡します．

```yaml
inputs:
  choice_data:
    kind: table_binding
    processor:
      id: davis/csv-transform
      version: 0.4.0
    sources:
      choices:
        kind: local
        path: choices.csv
      persons:
        kind: local
        path: persons.csv
    base: choices
    joins:
      - source: persons
        relationship: many_to_one
        left_on: case_id
        right_on: person_id
    columns:
      travel_time:
        source: choices
        column: time
      income:
        source: persons
        column: income
```

`columns`のkeyはモデル設定から参照する最終列名です．GUIではparameterごとにsourceとcolumnを選び，この対応を生成します．結合済み表はRunの`artifacts/prepared/`へ保存され，`result.json`の`prepared_input:<slot名>` extensionにpath，media type，size，BLAKE3が記録されます．前処理processのrequest，result，logも`preparation/`へ保存します．

現在のbinding joinはbase表の列と各追加sourceの列を照合します．単一keyと複合key，`many_to_one`と`one_to_one`，left／inner，未一致許可を指定できます．bindingの入れ子と，追加source同士を順に結ぶjoinは未対応です．完全なpipeline DAGとは別に，モデル入力を組み立てるための限定された一段前処理です．実動作例は[`components/davis-mnl/examples/multi-source`](../components/davis-mnl/examples/multi-source/)にあります．

## 参考transform component

[`components/davis-csv-transform`](../components/davis-csv-transform/)は，任意codeをYAMLへ埋め込まず，数値列の線形結合を新しいCSV列として作る参考実装です．

```console
cargo run -p davis-cli -- \
  --repository . \
  model run components/davis-csv-transform/examples/minimal/transform.yaml
```

`examples/mnl-chain`には，CSV変換Runの`transformed_table`をMNLへ渡す2段階exampleがあります．

## 参考Nested Logit component

[`components/davis-nl`](../components/davis-nl/)は，選択肢をnestへ分ける2段階Nested Logitの読みやすい参考実装です．MNLと共通の`roles`と`terms`に加えて，`nests`で各選択肢が所属するnestを1つずつ指定します．各nestの`dissimilarity`は`fixed`で固定するか，`initial`を初期値として推定できます．singleton nestは省略時に1へ固定します．

```yaml
nests:
  - name: motorized
    alternatives: [train, car]
    dissimilarity:
      initial: 0.8
  - name: active
    alternatives: [walk]
    dissimilarity:
      fixed: 1.0
```

全選択肢が重複なくいずれか1つのnestへ入る必要があります．推定する非類似度parameterは`0.05`から`1.0`へ制約されます．これはcross-nested logitではなく，2段階の非重複NLです．最小例は次で実行できます．

```console
cargo run -p davis-cli -- \
  --repository . \
  model run components/davis-nl/examples/minimal/model.yaml
```

### 複数CSVのjoin

`davis/csv-transform`は，`table`を基準表として任意名の追加CSV inputを受け取れます．join keyは単一列または複数列で指定します．次の例では，tripの`origin_zone`とzone表の`zone_code`が同じ行を結合します．

```yaml
inputs:
  table:
    kind: local
    path: trips.csv
  zones:
    kind: local
    path: zones.csv
config:
  joins:
    - input: zones
      how: left
      relationship: many_to_one
      left_on: origin_zone
      right_on: zone_code
      columns:
        origin_population: population
```

`columns`は「出力列名: join元の列名」の対応です．複合keyでは`left_on: [person_id, date]`のように配列を使います．`many_to_one`では右表のkey重複を拒否し，`one_to_one`では左右両方の重複を拒否します．一致しないkeyは既定でerrorです．意図的に許可する場合だけ`allow_unmatched: true`を指定し，`left` joinでは空欄を補い，`inner` joinでは該当行を除外します．

動作例は次で実行できます．

```console
cargo run -p davis-cli -- \
  --repository . \
  model run components/davis-csv-transform/examples/join/transform.yaml
```

### CSV／Parquet出力

既定では人間が確認しやすいCSVを出力します．容量，読込速度，列型の保持を優先する場合はParquetを指定できます．

```yaml
config:
  output:
    format: parquet
    compression: zstd
    null_values: [""]
    column_types:
      person_id: string
      travel_time: float64
```

`column_types`を省略した列は，`string`，`int64`，`float64`，`boolean`から安全側に型推定します．`001`のように先頭0がある整数風の値はIDとみなして`string`を維持します．曖昧さを避けたい場合は型を明示してください．実際の形式とParquet schemaは`transformation-summary.json`へ記録されます．`examples/mnl-chain`はParquetを出力し，そのartifactをMNL 0.2.0へ直接渡します．

## 検証と公開

```console
davis install component ./my-component
davis model validate analysis.yaml
davis model run analysis.yaml
davis component pack ./my-component --name my-component --out dist
davis component registry dist/my-component-0.1.0.entry.json \
  --out dist/component-registry.json
```

公開bundleには`.venv`，`__pycache__`，Git metadata，test cacheを含めません．dependencyをlockし，正常系，欠損列，不正値，成果物欠落をtestしてください．信頼できないcomponentは任意codeを実行できるため，現在はinstallしないでください．sandboxとregistry署名は未実装です．

## Davisを知らないAIによる最終検証

このframeworkの最終検証では，Davis開発の会話履歴やsource codeを見ていないブラウザAIを使います．AIへ渡してよいものは，このガイド，作りたいモデルの要件，入出力の小さなsampleだけです．既存componentを模写させることは必須にしません．

次をすべて満たした場合に合格とします．

1. AIが`component.yaml`と実programを作成できます．
2. `davis component validate`が成功します．
3. AIが作成したAnalysis planをGUIで開いて内容を理解・変更できます．
4. sample dataによる実行が成功し，宣言したartifactをDavisが表示できます．
5. 入力列不足等の失敗が，利用者の直せる説明として表示されます．
6. 別の人がManifest，plan，結果を読み，使用file，設定，出力を説明できます．

失敗した場合はAIだけを調整して通すのではなく，誤解された箇所をこのガイド，scaffold，validator，GUIのいずれかへ還元します．特定AIの事前知識に依存しないことが目的です．

## 現在の境界

実装済みなのはlocal input，`run_artifact` input，推定時の複数source binding，宣言的な複数CSV join，列選択，線形結合，CSV／Parquet出力，process実行，artifact検証，local／registry installです．catalog input，filter，group，任意pipeline DAG，sandboxは未実装です．DavisはPython等の一般言語環境をinstallしません．QGIS等の手作業は生成済みfileをlocal inputとして利用し，自動実行できるalgorithmは同じprocess contractでtransform componentとして包めます．
