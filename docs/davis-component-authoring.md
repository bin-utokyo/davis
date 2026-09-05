# Davis Component Authoring Guide

この文書は，人間とAIがDavisへ独自の計算処理を追加するための最小契約を示します．推定器だけでなく，入力CSVから説明変数CSVを作る処理や，結果を図表へ変換する処理も同じcomponentとして実装できます．

## 基本構造

すべてのcomponentは，入力参照，設定，process実装，成果物宣言を持ちます．Davisは入力解決，digest，process起動，log，成果物検証，実行記録を共通に担当します．

```text
component/
├── component-manifest.yaml
├── pyproject.toml
├── uv.lock
├── schemas/
│   ├── config.schema.json
│   └── ui.schema.json
├── src/
└── tests/
```

新規componentの正規形式は`component-manifest.yaml`と`davis.component/v1alpha1`です．旧形式の`model-manifest.yaml`と`davis.model/v1alpha1`も後方互換のため読み込めますが，1つのpackageへ新旧両方のManifestを置くことはできません．`kind`で役割を区別します．

## Manifest

```yaml
api_version: davis.component/v1alpha1
id: example/accessibility
name: Accessibility calculator
version: 0.1.0
kind: transform
requires_davis: ">=0.3.5"

runtime:
  kind: python
  command: ["uv", "run", "--frozen", "python", "-m", "accessibility"]
  request_argument: "--request"
  lockfile: uv.lock

operations: [transform]
inputs:
  - name: persons
    media_types: [text/csv]
    required: true

config_schema: schemas/config.schema.json
ui_schema: schemas/ui.schema.json

outputs:
  artifacts:
    explanatory_variables:
      media_types: [text/csv]
      required: true
```

`kind`は`model`，`transform`，`visualize`のいずれかです．省略時は既存Manifestとの互換性のため`model`です．`operations`の名前はcomponentが定義します．`requires_davis`はDavis本体のrelease versionから自動生成せず，利用するcontractの互換範囲をcomponent作者が宣言します．

通常は`inputs`でslot名を固定します．CSV joinのように利用者が任意名の追加inputを与えるcomponentだけは，`additional_inputs.media_types`で許可する形式を限定できます．追加inputもDavisがpath，media type，size，BLAKE3を解決・記録してからcomponentへ渡します．

`outputs.artifacts`を宣言したcomponentは，未宣言artifact，必須artifactの欠落，異なるmedia typeを返せません．Davisは各成果物のpath，size，BLAKE3も検証して`result.json`へ記録します．既存componentの`outputs.standard`と`outputs.extensions`は引き続き読めます．

## Analysis plan

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

DavisはManifestの`runtime.command`をcomponent directoryで起動し，`request_argument`の直後に`request.json`の絶対pathを渡します．componentは次を行います．

1. `request.json`を読みます．
2. `inputs.<name>.resolved.path`だけから入力を読みます．
3. `output_directory`以下へ成果物を書きます．
4. `output_directory/run-result.json`を書きます．
5. 成功時は終了code 0，失敗時は非0を返します．

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

## 参考transform component

[`components/davis-csv-transform`](../components/davis-csv-transform/)は，任意codeをYAMLへ埋め込まず，数値列の線形結合を新しいCSV列として作る参考実装です．

```console
cargo run -p davis-cli -- \
  --repository . \
  model run components/davis-csv-transform/examples/minimal/transform.yaml
```

`examples/mnl-chain`には，CSV変換Runの`transformed_table`をMNLへ渡す2段階exampleがあります．

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

## 現在の境界

実装済みなのはlocal input，`run_artifact` input，宣言的な複数CSV join，線形結合，process実行，artifact検証，local／registry installです．catalog input，filter，group，pipeline DAG，Davis管理Python，sandboxは未実装です．QGIS等の手作業は生成済みfileをlocal inputとして利用し，自動実行できるalgorithmは同じprocess contractでtransform componentとして包めます．
