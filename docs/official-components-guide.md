# Davis公式Component利用ガイド

[English](official-components-guide_en.md)

このガイドは，Davis v0.5.5で配布する4つの公式componentを使って，データ変換またはモデル推定を行う人向けです．新しいcomponentを作るための文書ではありません．Davisの開発repositoryや内部実装を知らない人，およびその人を支援するAIは，このガイドだけを公式componentの利用仕様として参照できます．

## 1．最初に理解すること

Davisでは，component本体と，1回ごとの分析条件を分けます．

- `component.yaml`：Davisが配布するcomponentの仕様です．通常の利用者は編集しません．
- `model.yaml`または任意名のAnalysis Plan：今回使うfile，列，説明変数，推定設定です．利用者またはAIが作成・編集します．
- `davis-runs/`：実行時にDavisが作る結果directoryです．

AIに支援を依頼する場合は，このガイド，実施したいモデルまたは変換内容，および利用する列を説明するschemaファイルを渡してください．schemaファイルには，実データを含めず，列名，データ型，単位，欠損値の扱い，カテゴリ値の意味等だけを記載します．schemaを用意できない場合は，実データから切り離したheaderだけを渡し，列名自体が機密情報を含み得る場合は匿名化した列名と意味の対応表を使用してください．CSV本体，実際の値，個人名，ID等の機密情報はAIへ渡さないでください．AIは`component.yaml`を変更せず，Analysis Planを作成してください．

## 2．必要な環境

Davis CLIをv0.5.5へ更新します．

```console
davis update
davis --version
```

公式componentは`uv 0.8`以上でPython環境と固定済みdependencyを起動します．Davisは`uv`自体をinstallしません．

```console
uv --version
```

`uv`がない場合は，[uv公式導入手順](https://docs.astral.sh/uv/getting-started/installation/)を参照してください．

## 3．公式componentのinstall

```console
davis install component mnl
davis install component nl
davis install component rl
davis install component csv-transform
davis installed
```

| 名前 | ID | Version | 目的 |
| --- | --- | --- | --- |
| Multinomial Logit | `davis/mnl` | `0.3.2` | 選択肢long形式データからMNLを推定 |
| Nested Logit | `davis/nl` | `0.1.2` | 2段階・非重複nestのNLを推定 |
| Recursive Logit | `davis/rl` | `0.1.2` | link networkと観測経路からRLを推定 |
| CSV Transform | `davis/csv-transform` | `0.4.1` | CSV結合，線形結合列，列選択，CSV／Parquet出力 |

複数CSVをモデル入力として結合する場合，Desktopは内部で`davis/csv-transform`を使用するため，対象モデルとCSV Transformの両方をinstallしてください．

## 4．Desktopで使う共通手順

```console
davis install desktop
davis desktop
```

1. `Project workspace`で空または既存の作業folderを選びます．Git repositoryである必要はありません．
2. `ComponentManifest`で使用するcomponentを選びます．
3. 入力欄から`ローカルCSV`または`Davis Catalog`を選びます．
4. 複数CSVを追加した場合は，基準データ，左右の結合key，関係，left／inner，未照合を許可するかを指定します．
5. 役割列，説明変数，モデル固有設定を入力します．
6. `YAMLを確認`でAnalysis Planを確認し，保存して実行します．
7. 結果は画面と`<workspace>/davis-runs/<run-id>/`の両方に残ります．

入力CSVはworkspace外にあっても選択できます．Catalog入力は`dataset_id`と`file_id`としてPlanへ記録され，共通cacheから解決されます．

## 5．CLIでAnalysis Planを実行する

GUIで保存したPlanも，AIまたは人が書いたPlanも，同じcommandで検証・実行できます．

```console
davis model validate ./model.yaml
davis model plan ./model.yaml
davis model run ./model.yaml
```

既定では，current directoryの`davis-runs/`へ結果を保存します．別の場所に保存する場合は`--run-root`を指定します．Plan内のlocal pathはPlan fileがあるdirectoryを基準に解決されます．

## 6．MNL

### 入力

`choice_data`へCSVまたはParquetを1つ指定します．1行が「1ケース内の1選択肢」を表すlong形式にしてください．

最低限必要な役割は次のとおりです．

| Role | 意味 |
| --- | --- |
| `case_id` | 選択状況を識別する1列または複数列 |
| `alternative_id` | その行の選択肢ID |
| `chosen` | 選ばれた行を示す0／1等の列 |
| `chosen_alternative` | ケースごとに選択された選択肢IDを持つ列．`chosen`の代わりに使用可能 |
| `available` | 利用可能性を表す列．任意 |
| `weight` | ケースの重み．任意 |

`chosen`と`chosen_alternative`はどちらか一方を指定します．

### 効用term

各termにはparameter名と，入力列または定数のどちらかを指定します．`alternatives`を省略すると全選択肢へ適用し，指定すると該当選択肢だけへ適用します．

```yaml
terms:
  - parameter: beta_time
    column: time
  - parameter: beta_cost
    column: cost
    alternatives: [train, car]
  - parameter: asc_car
    constant: 1
    alternatives: [car]
```

推定設定では`optimizer` (`bfgs`または`l-bfgs-b`)，`max_iterations`，`tolerance`を指定できます．`development_case_limit`は動作確認用であり，本推定では省略します．

### 最小Plan

```yaml
api_version: davis.analysis/v1alpha1
name: mode-choice-mnl
component: {id: davis/mnl, version: 0.3.2, operation: estimate}
inputs:
  choice_data: {kind: local, path: choice.csv}
config:
  roles:
    case_id: case_id
    alternative_id: alternative
    chosen: chosen
  terms:
    - {parameter: beta_time, column: time}
    - {parameter: asc_car, constant: 1, alternatives: [car]}
  estimation: {optimizer: bfgs, max_iterations: 500, tolerance: 1.0e-8}
run: {label: mnl-baseline, tags: [mnl, baseline]}
```

## 7．Nested Logit

入力long形式，`roles`，`terms`はMNLとほぼ同じですが，`chosen`列が必須で，`nests`も指定します．全選択肢は重複なく1つのnestへ所属させます．

このcomponentは最上位scaleを1へ正規化し，各nestの非類似度`dissimilarity`をλとして扱います．λは`0 < λ <= 1`です．`initial`は推定初期値，`fixed`は固定値です．両方を同時に書きません．選択肢が1つだけのnestは実行時にλ=1へ固定されます．

```yaml
api_version: davis.analysis/v1alpha1
name: mode-choice-nl
component: {id: davis/nl, version: 0.1.2, operation: estimate}
inputs:
  choice_data: {kind: local, path: choice.csv}
config:
  roles: {case_id: case_id, alternative_id: alternative, chosen: chosen}
  terms:
    - {parameter: beta_time, column: time}
    - {parameter: asc_car, constant: 1, alternatives: [car]}
  nests:
    - name: motorized
      alternatives: [train, car]
      dissimilarity: {initial: 0.8}
    - name: active
      alternatives: [walk]
      dissimilarity: {fixed: 1.0}
  estimation: {max_iterations: 500, tolerance: 1.0e-8}
run: {label: nl-baseline, tags: [nl, baseline]}
```

## 8．Recursive Logit

RLは2つの表を受け取ります．

- `network`：有向linkを1行ずつ持つ表
- `observations`：tripごとの実際の通過linkを順番に並べたlong形式の表

Networkの必須roleは`link_id`，`from_node`，`to_node`です．観測経路の必須roleは`trip_id`，`step`，`link_id`，`destination`です．`step`はtrip内の通過順を一意に並べられる値にします．`destination`は目的地node IDです．両表のlink IDとnode IDは一致させてください．

効用termはnetwork表の数値列を参照します．`coefficient`は通常1で，変数の符号や倍率を事前に固定したい場合だけ変更します．parameterごとに`initial`，`lower`，`upper`を指定できます．

```yaml
api_version: davis.analysis/v1alpha1
name: route-choice-rl
component: {id: davis/rl, version: 0.1.2, operation: estimate}
inputs:
  network: {kind: local, path: network.csv}
  observations: {kind: local, path: observations.csv}
config:
  network_roles: {link_id: link_id, from_node: from_node, to_node: to_node}
  observation_roles: {trip_id: trip_id, step: step, link_id: link_id, destination: destination}
  terms:
    - {parameter: beta_time, column: time, coefficient: 1}
    - {parameter: beta_toll, column: toll, coefficient: 1}
  parameters:
    beta_time: {initial: -1.0, lower: -10.0, upper: -0.000001}
    beta_toll: {initial: -0.5, lower: -10.0, upper: 0.0}
  estimation: {max_iterations: 500, tolerance: 1.0e-8}
run: {label: rl-baseline, tags: [rl, baseline]}
```

## 9．CSV Transform

CSV Transformは推定器ではなく，入力データを再現可能な規則で加工するcomponentです．次の操作を組み合わせられます．

- `joins`：単一keyまたは複合keyによる追加CSVのleft／inner join
- `calculations`：列と定数の線形結合
- `select`：最終列の選択とrename
- `output`：CSVまたはParquetと圧縮方式の指定

結合の`relationship`は`many_to_one`または`one_to_one`です．想定外の重複はerrorになります．`allow_unmatched`の既定値は`false`です．

```yaml
api_version: davis.analysis/v1alpha1
name: prepare-choice-data
component: {id: davis/csv-transform, version: 0.4.1, operation: transform}
inputs:
  table: {kind: local, path: choices.csv}
  persons: {kind: local, path: persons.csv}
config:
  joins:
    - input: persons
      how: left
      relationship: many_to_one
      left_on: person_id
      right_on: person_id
      allow_unmatched: false
      columns: {income: income}
  calculations:
    - output: income_cost
      operation: linear_combination
      terms:
        - {column: income, coefficient: 0.001}
        - {column: cost, coefficient: -1}
  select:
    columns:
      case_id: case_id
      alternative: alternative
      chosen: chosen
      income_cost: income_cost
  output: {format: parquet, compression: zstd}
run: {label: prepare-choice-data, tags: [transform]}
```

CSV Transformを別Runとして先に実行する以外に，Desktopで複数CSVをモデル入力へ追加し，推定直前の共通`table_binding`として実行する方法もあります．

## 10．結果と確認箇所

モデルcomponentは，利用可能な場合に次の成果物を返します．

- `parameters.csv`：parameter名，推定値，標準誤差等
- `metrics.json`：対数尤度，AIC，BIC，収束情報等
- `predictions.csv`：予測確率または観測link選択確率
- `sample-summary.json`：利用ケース数，除外数，警告等

CSV Transformは`transformed.csv`または`transformed.parquet`と，変換summaryを返します．実際の成果物一覧は各Runの`result.json`を正とします．

error時は，まず次を確認してください．

1. `davis --version`が0.5.5以上か．
2. `uv --version`が0.8以上か．
3. `davis installed`に必要なcomponentがあるか．
4. CSV headerとPlanの列名が完全に一致しているか．
5. caseごとの選択結果，join key，networkと観測経路のIDが整合しているか．

独自componentを新しく作る場合は，[Davis Component Authoring Guide](davis-component-authoring.md)を参照してください．
