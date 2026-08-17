# Davis交通行動モデル統合プラットフォーム構想

> 状態: Draft 0.1
>
> 作成日: 2026-08-17
>
> 対象: Davisをデータ配布ツールから，交通行動モデルの準備・推定・理解を一体化したプラットフォームへ発展させる計画
>
> 原則: 未確定事項は推測で確定せず，本書内で「要確認」と明示する

## 1. エグゼクティブサマリー

Davisの目標を，次の1文に定めます．

> 学術的な行動モデルの知識や複雑な実行環境の構築なしに，利用者がデータを選び，整形し，モデルを指定し，推定結果を理解できるオープンな交通行動モデル基盤を提供する．

短期MVPでは，すべてを一度に作りません．次の縦方向の一連の体験を最優先にします．

1. Webでデータカタログを閲覧する
2. データセットの説明と列定義を確認する
3. サンプルデータまたはアップロードデータからMNLを設定する
4. 推定ジョブを実行する
5. 係数，標準誤差，適合度，警告を表と基本グラフで確認する
6. 入力，モデル設定，推定結果を再現可能な成果物として取得する

推奨する技術の中心はRustです．`davis_core`，`davis_api`，`davis_mnl`，`davis_fmt`をRustで実装し，Web UIはTypeScript，PCアプリは同じWeb UIを再利用するTauriとします．可視化はVega-Liteを第一候補とし，PythonとMatplotlibを必須依存にしません．Pythonは既存モデルや研究者独自コードを接続するための任意アダプターとして残します．

ストレージはCloudflare R2を第一候補としながら，コード上ではS3互換APIやローカルファイルへ差し替え可能にします．DVCはサーバーや利用者の必須ランタイムから外し，既存データの移行，管理者向け同期，研究用再現パイプラインに限定して残す方針を推奨します．

## 2. 目標と成功条件

### 2.1. プロダクト目標

- 初学者が画面上の説明に従い，MNL推定を完了できる
- 研究者が同じ基盤へ独自モデル，整形処理，可視化を追加できる
- CLI，Web，PCアプリから同じ概念と同じ結果を利用できる
- データの意味，出典，ライセンス，列定義，バージョンを実データと一緒に配布できる
- すべての推定結果について，入力データの版，整形手順，モデル仕様，実装の版を追跡できる
- R2，S3，ローカルファイルなどの保存先を，ドメインロジックを書き換えずに変更できる

### 2.2. MVPの成功条件

- 代表的な1データセットをカタログ表示・ダウンロードできる
- CSVまたはParquetを読み込み，MNL用long形式として検証できる
- 選択肢，選択列，説明変数，基準選択肢をGUIから指定できる
- MNLの最尤推定が完了し，収束判定と基本統計量を返せる
- 同一入力と同一設定から同一結果を再実行できる
- エラー時に，初学者が次に直す箇所を日本語で理解できる
- Web UIとCLIが同じAPI契約および結果スキーマを利用する

### 2.3. 初期段階で対象外とするもの

- あらゆる離散選択モデルへの対応
- 大規模な分散計算基盤
- ノーコードでの任意モデル生成
- 高度なGIS編集機能
- リアルタイム交通シミュレーション
- 複数組織向けの複雑な権限管理
- PCアプリの初回MVP同時リリース

## 3. 現状評価

現在の`packages/dataset_cli`は，次の資産をすでに持っています．

- データセット一覧，情報表示，ダウンロードのCLI
- PydanticによるデータセットYAMLとリリースマニフェストのスキーマ
- CSV，Excel，JSON，Parquetからのスキーマ推測
- 日本語・英語のメタデータ
- DVCとGoogle Driveによるデータ取得
- GitHub Releaseからのマニフェスト配布

一方，統合プラットフォームへ発展させる際には次の制約があります．

- CLIがDVC，Google Drive，PDF生成，UI表示，スキーマ処理を直接抱えている
- Python 3.13以上と多数の依存関係を利用者環境へ要求している
- カタログ，実データ取得，認証がGoogle Driveの構成へ結合している
- 既存列スキーマはデータ型の記述が中心で，行動モデル上の意味や整形履歴を表現できない
- 既存MNLは特定の入力ファイル構造とPythonコードに結合し，結果が機械可読な共通契約になっていない
- Web，デスクトップ，CLIで再利用できるアプリケーション境界がまだない

既存コードは破棄せず，MVPの移行元と回帰試験の基準として利用します．

## 4. 設計原則

1. **契約を先に作る**: UIやストレージ実装より先に，データ，モデル要求，結果，ジョブのスキーマを定義します．
2. **ドメインと配信方法を分ける**: `davis_core`はHTTP，R2，画面表示を知りません．
3. **ライブラリAPIとHTTP APIを対応させる**: ローカルCLIとサーバーの挙動差を最小化します．
4. **データは自己記述的にする**: 実データ，`dataset.yaml`，チェックサム，ライセンスを同じリリース単位で扱います．
5. **不変な版を参照する**: `latest`を保存せず，データ版と内容ハッシュを推定記録へ固定します．
6. **推定と可視化を分ける**: 推定器は構造化結果を返し，可視化側はその結果から表示を構成します．
7. **拡張は言語ABIではなくデータ契約で行う**: 独自モデルをRustの動的ライブラリABIへ固定しません．
8. **初学者向け説明と研究者向け詳細を両立する**: 画面には平易な説明を出し，詳細ログと数値は失わないようにします．
9. **秘密情報をクライアントへ配らない**: ストレージ資格情報はサーバー側だけに置き，短時間の署名付きURLを発行します．
10. **Pythonは任意にする**: 標準経路はPythonなしで動作し，必要な研究コードだけを隔離されたランナーで実行します．

## 5. 推奨アーキテクチャ

### 5.1. 全体像

```mermaid
flowchart LR
    Web["davis_web<br/>ブラウザUI"]
    App["davis_app<br/>Tauri PCアプリ"]
    CLI["davis_cli<br/>ローカル／遠隔操作"]
    API["davis_api<br/>HTTP・認証・ジョブ管理"]
    Core["davis_core<br/>ユースケース・ドメイン"]
    Contracts["davis_contracts<br/>共通スキーマ"]
    Catalog["davis_catalog<br/>データ目録"]
    Fmt["davis_fmt<br/>検証・整形"]
    MNL["davis_mnl<br/>標準MNLプラグイン"]
    Viz["davis_viz<br/>可視化仕様生成"]
    Runner["davis_runner<br/>モデル実行境界"]
    Store["ObjectStore<br/>R2／S3／ローカル"]
    Meta["Metadata DB<br/>SQLite／PostgreSQL"]

    Web --> API
    App --> API
    App -. "オフライン時" .-> Core
    CLI --> API
    CLI -. "ローカル時" .-> Core
    API --> Core
    Core --> Catalog
    Core --> Fmt
    Core --> Runner
    Runner --> MNL
    Core --> Viz
    Catalog --> Store
    Core --> Store
    API --> Meta
    Contracts --- Web
    Contracts --- App
    Contracts --- CLI
    Contracts --- Core
    Contracts --- MNL
    Contracts --- Viz
```

### 5.2. コンポーネント責務

| コンポーネント | 責務 | 単独利用 | MVP |
| --- | --- | --- | --- |
| `davis_contracts` | JSON Schema，Rust型，OpenAPIの共通契約 | 可 | P0 |
| `davis_core` | カタログ取得，データ取得，整形，推定，成果物管理のユースケース | 可 | P0 |
| `davis_api` | HTTP，認証，署名付きURL，非同期ジョブ，エラー変換 | サービスとして可 | P0 |
| `davis_catalog` | `dataset.yaml`の検証，索引，検索，版管理 | 可 | P0 |
| `davis_mnl` | 一般的なMNL仕様の検証，推定，予測，共通結果の出力 | 可 | P0 |
| `davis_web` | データ閲覧，モデル設定ウィザード，結果表示 | Webとして可 | P0 |
| `davis_fmt` | 生データの読込，標準化，欠損処理，変換レシピ | 可 | P1の最小部のみP0 |
| `davis_viz` | 共通結果からVega-Lite仕様と表を生成 | 可 | P1の最小部のみP0 |
| `davis_cli` | 上記ユースケースの薄いCLIクライアント | 可 | P1．既存CLIは当面維持 |
| `davis_app` | Web UIを再利用したTauriデスクトップアプリ | 可 | P2 |
| `davis_runner` | 標準モデルと外部モデルの安全な実行，進捗，取消 | 可 | P1．P0はプロセス内実行 |

### 5.3. `davis_core`の位置付け

`davis_core`を「ストレージそのもの」にすると，R2やDVCの都合が全機能へ波及します．そのため，`davis_core`はアプリケーションの根幹となるドメインとユースケースを持ち，ストレージは次のような抽象インターフェースとして扱う設計を推奨します．

```rust
#[async_trait]
pub trait ArtifactStore {
    async fn get(&self, key: &ArtifactKey) -> Result<ByteStream, StoreError>;
    async fn put(&self, request: PutArtifact) -> Result<ArtifactRef, StoreError>;
    async fn stat(&self, key: &ArtifactKey) -> Result<ArtifactMeta, StoreError>;
    async fn signed_get_url(&self, key: &ArtifactKey, ttl: Duration)
        -> Result<Url, StoreError>;
}
```

R2，S3，ローカルファイルはこのインターフェースのアダプターです．カタログ検索用のメタデータはオブジェクトストレージの一覧走査に頼らず，SQLiteまたはPostgreSQLの読み取りモデルへ索引化します．

## 6. 共通契約

### 6.1. 契約の配布

- HTTP API: OpenAPI 3.1
- 永続化・プラグイン契約: JSON Schema
- 表データの保存: Parquet
- 高速なプロセス間転送: Arrow IPC
- 人が編集する設定: YAML
- Web表示用の小さな応答: JSON
- 大きなファイル: APIを経由せず，短時間の署名付きURLで直接転送

YAMLとJSONを別々の仕様にしません．JSON Schemaを規範とし，YAMLはその表現形式として検証します．Rust型から生成する場合も，CIで生成差分を検出して契約変更をレビューします．

### 6.2. バージョニング

- HTTPは`/api/v1`のようにメジャーバージョンをURLへ含める
- 各スキーマは`schema_version: 1`を持つ
- 後方互換なフィールド追加は同一メジャー版で許可する
- 既存フィールドの意味変更，削除，型変更は次のメジャー版で行う
- データセット版はSemVerを強制せず，発行者が付けた不変の`version`と`sha256`で特定する
- 推定結果は使用した全入力の内容ハッシュと実装版を記録する

### 6.3. 共通エラー

すべての入口で次の構造を返します．HTTPではRFC 9457形式のProblem Detailsへ対応させます．

```json
{
  "type": "https://davis.example/errors/invalid-choice-column",
  "title": "選択結果の列を利用できません",
  "status": 422,
  "code": "DAVIS-MNL-1003",
  "detail": "choice列には各case_id内で1つだけtrueが必要です",
  "instance": "/api/v1/jobs/job_01...",
  "issues": [
    {"row": 18, "column": "choice", "hint": "case_id=trip-7を確認してください"}
  ]
}
```

`code`は安定させ，`detail`は日本語と英語へローカライズできるようにします．

### 6.4. 非同期ジョブ

データ整形や推定は，短い処理でも最初からジョブとして表現します．MVPでは単一プロセスのインメモリキューでも構いませんが，契約は次の状態機械を守ります．

```text
queued -> running -> succeeded
                  -> failed
                  -> cancelling -> cancelled
```

ジョブには`progress`，`stage`，`warnings`，`created_at`，`started_at`，`finished_at`，`result_ref`を持たせます．これにより，将来のワーカー分離や大規模計算へ移行できます．

## 7. データカタログと保存方式

### 7.1. 推奨オブジェクト構造

```text
datasets/
  jp-pt-tokyo/
    2008-r1/
      dataset.yaml
      files/
        trips.parquet
        zones.parquet
      docs/
        README.ja.md
        README.en.md
      checksums.sha256
catalog/
  v1/
    index.json
artifacts/
  sha256/
    ab/cd/<full-digest>
runs/
  <run-id>/
    request.json
    result.json
    diagnostics.json
    charts.json
```

公開済みの`datasets/<id>/<version>/`は上書き禁止とします．修正時は新しい版を発行します．`artifacts/sha256/`は重複排除と再現性のための内容アドレス領域です．

### 7.2. `dataset.yaml`の役割

既存の列名と型に加えて，モデルや整形が理解できる意味情報を持たせます．ただし，すべてのデータに一つの物理列名を強制しません．元の列名を保存しつつ，`semantic_role`と変換レシピで共通概念へ対応付けます．

```yaml
schema_version: 1
id: jp-pt-tokyo
version: 2008-r1
title:
  ja: 東京都市圏PT調査
  en: Tokyo Metropolitan Area Person Trip Survey
description:
  ja: 東京都市圏のパーソントリップ調査データ
  en: Person trip survey data for the Tokyo metropolitan area
license:
  id: LicenseRef-Proprietary
  text:
    ja: 利用条件をここに記載
    en: Usage terms go here
access:
  level: restricted
provenance:
  publisher: example-publisher
  source_url: https://example.invalid/source
  issued_at: 2026-08-17
tables:
  - id: trips
    path: files/trips.parquet
    media_type: application/vnd.apache.parquet
    sha256: "<64桁のハッシュ>"
    row_count: 1000
    columns:
      - name: TripID
        data_type: int64
        nullable: false
        semantic_role: trip.id
      - name: Transportation
        data_type: int64
        nullable: false
        semantic_role: choice.mode
        code_list:
          "1": rail
          "2": bus
```

必要項目を増やし過ぎると登録が止まるため，必須項目は`id`，`version`，表示名，ライセンス，アクセス区分，ファイルのパス・型・ハッシュに絞ります．列説明や意味役割は段階的に充実させます．

### 7.3. YAMLの正本と公開

MVPでは，レビュー可能な`catalog/datasets/<id>/<version>/dataset.yaml`をGit上の正本とします．公開ジョブが検証後に実データと同じR2プレフィックスへ同一内容を配置し，`catalog/v1/index.json`を再生成します．Webは高速な`index.json`またはAPIを読み，詳細画面では`dataset.yaml`相当の情報を表示します．

将来，管理画面から登録できるようにする場合も，サーバーがデータとYAMLを一つの発行トランザクションとして扱い，Gitへのエクスポートまたは監査ログを残します．

### 7.4. R2とDVCの判断

| 選択肢 | 利点 | 課題 | 推奨用途 |
| --- | --- | --- | --- |
| R2をアプリから直接利用 | S3互換，署名付きURL，Web配信に適する，クライアントにDVC不要 | カタログと版管理はDavis側で設計が必要 | 本番ランタイムの第一候補 |
| R2をDVCリモートとして利用 | 既存DVC運用を移行しやすい，研究者のGit連携を維持 | Python／DVC依存，Web配信APIとしては扱いにくい | 管理者同期と移行 |
| Google Drive＋DVCを維持 | 現行資産をそのまま利用可能 | 利用者ごとのOAuth設定，アプリ境界への強い結合 | 移行期間のみ |
| S3／MinIO | 標準性が高く，オンプレミスも可能 | 運用費と構築作業が増える場合がある | 組織要件に応じた代替 |

結論として，DVCを全面廃止する必要はありませんが，Davisの公開APIと利用者体験の必須要素にはしません．DVCのS3互換エンドポイント設定を利用すればR2を管理用リモートとして検証できます．一方，WebとPCアプリのダウンロードは`davis_api`が発行する署名付きURLを使います．

## 8. `davis_fmt`の設計

### 8.1. 「一般データ入力」の定義

一般入力を特定のDataFrameライブラリのオブジェクトにしません．境界では次を標準とします．

- 保存と交換: Parquet＋Arrowスキーマ
- 少量の利用者入力: CSV，XLSX，JSONを受け付け，直ちに内部Parquetへ正規化
- 実行時: RustのArrow／Polars表現
- 外部プラグイン: Parquetファイル参照またはArrow IPCストリーム

これにより，Pythonのpandas，R，Julia，Goなども同じデータへ接続できます．Apache Arrowは言語非依存の列指向フォーマットであるため，特定言語のDataFrame APIを共通契約にするより安定します．

### 8.2. 変換レシピ

整形はコードだけでなく，機械可読かつ再実行可能なレシピとして保存します．

```yaml
schema_version: 1
input_table: raw_trips
output_table: mnl_long
steps:
  - op: rename
    columns:
      TripID: case_id
      Transportation: chosen_mode
  - op: map_values
    column: chosen_mode
    values:
      "1": rail
      "2": bus
  - op: cast
    column: travel_time
    to: float64
  - op: filter
    expression: travel_time >= 0
```

初期対応する演算は，列名変更，型変換，値対応，列選択，欠損除外，単純な式による列生成，wide／long変換に限定します．任意コード実行はP0へ含めません．

### 8.3. 列名の標準化

「一般的な名称」には唯一の世界標準がないため，物理列名を一括変更する設計は避けます．次の3層で管理します．

1. `source_name`: 元データの列名
2. `canonical_name`: 当該レシピ内の安定した英語snake_case名
3. `semantic_role`: `trip.id`，`choice.selected`，`level_of_service.travel_time`などの意味識別子

これにより，データ提供者の原本を壊さず，モデルごとに必要な役割を検証できます．

## 9. `davis_mnl`の設計

### 9.1. 入力形式

標準MNLはlong形式を規範入力とします．1行は「1ケースにおける1選択肢」を表します．最低限，次の役割が必要です．

| 役割 | 説明 | 例 |
| --- | --- | --- |
| `case_id` | 意思決定機会の識別子 | trip_001 |
| `alternative_id` | 選択肢 | rail |
| `choice` | 実際に選ばれたか | true |
| `available` | 選択可能か | true |
| 説明変数 | 時間，費用，個人属性など | 32.5 |

各`case_id`で`choice=true`がちょうど1件であること，選択された選択肢が利用可能であること，必要変数が数値であることを推定前に検証します．wide形式は`davis_fmt`が明示的なレシピでlong形式へ変換します．

### 9.2. モデル仕様

モデルの内部コードを書き換えなくても，一般的な効用関数を設定できる宣言形式を先に提供します．高度な研究モデルだけをプラグインとして追加します．

```yaml
schema_version: 1
model_type: mnl
data:
  table: mnl_long
  case_id: case_id
  alternative_id: alternative_id
  choice: choice
  availability: available
reference_alternative: walk
utilities:
  rail: asc_rail + beta_time * travel_time + beta_cost * cost
  bus: asc_bus + beta_time * travel_time + beta_cost * cost
  walk: beta_time * travel_time
parameters:
  - name: asc_rail
    initial: 0.0
  - name: asc_bus
    initial: 0.0
  - name: beta_time
    initial: -0.1
  - name: beta_cost
    initial: -0.01
optimizer:
  method: lbfgs
  max_iterations: 500
  tolerance: 1.0e-8
```

式言語は四則演算，括弧，列参照，パラメーター参照から開始し，任意関数や任意コードは許可しません．式を構文木へ変換してデザイン行列を構築すれば，GUIの式ビルダーとテキスト編集の両方を提供できます．

### 9.3. 共通推定結果

`davis_mnl`はHTMLや画像ではなく，次の共通構造を返します．

```json
{
  "schema_version": 1,
  "run_id": "run_01...",
  "model": {"id": "davis.mnl", "version": "0.1.0"},
  "status": "converged",
  "parameters": [
    {
      "name": "beta_time",
      "estimate": -0.083,
      "std_error": 0.012,
      "z_value": -6.92,
      "p_value": 4.5e-12,
      "confidence_interval_95": [-0.107, -0.060]
    }
  ],
  "fit": {
    "n_cases": 1000,
    "n_observations": 3000,
    "log_likelihood": -721.4,
    "null_log_likelihood": -1098.6,
    "rho_squared": 0.343,
    "adjusted_rho_squared": 0.339,
    "aic": 1450.8,
    "bic": 1470.4
  },
  "diagnostics": {
    "iterations": 28,
    "gradient_norm": 2.1e-7,
    "hessian_condition": 184.2,
    "warnings": []
  },
  "provenance": {
    "dataset_sha256": "<hash>",
    "model_spec_sha256": "<hash>",
    "software_revision": "<git-sha>"
  }
}
```

数値は形式例であり，実データの結果ではありません．推定器を書き換えてもこの共通部分を返せば，`davis_viz`はそのモデルを表示できます．モデル固有情報は`extensions`名前空間へ追加します．

### 9.4. 統計的な最低品質

P0であっても，係数だけを出して完了とはしません．最低限，次を実装・検証します．

- log-sum-expによる数値安定化
- 解析勾配または自動微分と有限差分による照合
- 収束判定と未収束警告
- Hessianまたは情報行列からの分散共分散
- 標準誤差，z値，p値，95%信頼区間
- 対数尤度，null対数尤度，McFaddenのρ²，補正ρ²，AIC，BIC
- 完全共線性，識別不能，利用不能な選択肢，欠損，オーバーフローの診断
- 小さな既知データと既存実装との数値比較

ロバスト標準誤差，クラスタリング，重み付き尤度，パネル構造はP1以降に分離します．

## 10. モデル差し替えとプラグイン境界

Rustの`trait`は同一ビルド内の拡張には有効ですが，第三者が別バージョンでビルドした動的ライブラリの安定ABIとしては使いません．拡張レベルを次の3段階にします．

1. **モデル設定**: 標準MNLのYAML式を変更する．大半の利用者はここで完結する
2. **同一ワークスペース実装**: Rustの`ModelEngine` traitを実装してDavis本体と一緒にビルドする
3. **外部プラグイン**: 実行可能ファイルまたはOCIコンテナが，バージョン付きJSON＋Parquet契約で入出力する

外部プラグインの例です．

```yaml
schema_version: 1
id: example.nested-logit
version: 0.1.0
protocol: davis-model-runner-v1
command: ["./davis-nl-plugin"]
capabilities:
  model_types: ["nested_logit"]
  operations: ["validate", "estimate", "predict"]
input_contract: davis://schemas/estimate-request/1
output_contract: davis://schemas/estimate-result/1
```

MVPでは標準MNLをプロセス内実行し，契約だけを外部実行可能な形にします．外部プロセス，OCI，WASIの実行はP1以降です．第三者コードをサーバー上で動かす際は，CPU，メモリ，時間，ネットワーク，ファイルアクセスを制限します．

## 11. `davis_viz`の設計

`davis_viz`はモデルの生データ構造を直接読まず，共通`EstimateResult`と追加の`PredictionResult`を読みます．出力は次の組合せです．

- 表示用ViewModel
- Vega-Lite JSON仕様
- CSV／JSONの表
- 必要に応じてSVG／PNGへ変換するエクスポート処理

P0の画面は次に限定します．

- 係数，標準誤差，p値，信頼区間の表
- 係数と95%信頼区間のフォレストプロット
- 適合度と収束状態のカード
- 推定警告と，平易な説明

P1では予測シェア，弾力性，限界効果，観測対予測，シナリオ比較を追加します．Vega-Liteは宣言的なJSON仕様でWebへ埋め込めるため，共通結果から図を生成する境界に適しています．Matplotlibは論文用図や既存研究コード向けの任意エクスポーターとして追加可能です．

## 12. WebとPCアプリ

### 12.1. `davis_web`

推奨構成はTypeScript＋React系フレームワークです．MVPではフレームワーク固有のサーバー機能へビジネスロジックを置かず，`davis_api`のクライアントとして実装します．OpenAPIからTypeScript型とAPIクライアントを生成します．

初学者向けの主要画面は次のとおりです．

1. データカタログ
2. データセット詳細とライセンス確認
3. データプレビューと品質チェック
4. 「何を選んだか」「選択肢は何か」から始まるモデル設定ウィザード
5. 推定前チェックと推定実行
6. 結果の要約，詳細，警告，ダウンロード

専門用語はツールチップだけに隠さず，「この数値から何が分かるか」と「何は断定できないか」を結果画面へ表示します．

### 12.2. `davis_app`

Tauriを利用し，`davis_web`のUIコンポーネントとAPIクライアントを再利用します．オンラインモードでは`davis_api`へ接続し，将来のオフラインモードではRustの`davis_core`を同梱します．

PCアプリをP2にする理由は，WebのUIと契約が固まる前にパッケージング，署名，自動更新，OS別試験へ時間を使わないためです．TauriはRust側とWebView側をメッセージングで接続できるため，Web完成後の再利用に向いています．

### 12.3. `davis_cli`

既存CLIは急いで全面移植せず，互換期間を設けます．新CLIは薄いクライアントとし，次の2モードを同じコマンド体系で提供します．

```text
davis catalog list
davis dataset get jp-pt-tokyo@2008-r1
davis fmt apply recipe.yaml --input raw.csv --output clean.parquet
davis mnl fit model.yaml --data clean.parquet --output run/
davis --server https://davis.example ...
```

- ローカルモード: `davis_core`を直接呼ぶ
- サーバーモード: `davis_api`を呼ぶ

## 13. HTTP APIの初期案

```text
GET    /api/v1/health
GET    /api/v1/datasets
GET    /api/v1/datasets/{dataset_id}/versions/{version}
POST   /api/v1/datasets/{dataset_id}/versions/{version}/download-url
POST   /api/v1/uploads
POST   /api/v1/uploads/{upload_id}/parts
POST   /api/v1/uploads/{upload_id}/complete
POST   /api/v1/format/validate
POST   /api/v1/format/jobs
POST   /api/v1/models/mnl/validate
POST   /api/v1/estimate/jobs
GET    /api/v1/jobs/{job_id}
POST   /api/v1/jobs/{job_id}/cancel
GET    /api/v1/runs/{run_id}
GET    /api/v1/runs/{run_id}/result
GET    /api/v1/runs/{run_id}/visualizations
```

`POST /estimate/jobs`はファイル本体を受け取らず，不変な`artifact_ref`と`model_spec`を受け取ります．同じ内容ハッシュと設定ハッシュの組合せは，再利用可能な結果としてキャッシュできます．

## 14. 推奨技術スタック

| 領域 | 第一候補 | 理由 | 代替 |
| --- | --- | --- | --- |
| Core／API／MNL／fmt | Rust | 単一バイナリ，型安全，Pythonランタイム不要，Polars／Arrow利用 | GoはAPI開発が速いが，表処理と数値計算を別実装にしやすい |
| HTTP | Axum系 | Tokioと統合しやすく，Rustの標準的構成 | Actix Web |
| 表処理 | Arrow＋Polars Rust | CSV／Parquet処理と列演算をRustで統一 | DataFusionを大規模SQL時に追加 |
| 数値計算 | nalgebraまたはndarray＋十分に検証した最適化器 | MNLの行列演算と最適化をPythonなしで実行 | 初期検証だけPython参照実装と比較 |
| Web | TypeScript＋React系 | UI資産，OpenAPI型生成，Vega-Lite連携 | Svelte系 |
| 可視化 | Vega-Lite | JSON契約，Web向け，宣言的 | Plotly.js，Observable Plot |
| PC | Tauri | Web UI再利用，Rust統合，OSのWebView利用 | Electron |
| オブジェクト保存 | R2＋Rust `object_store`抽象 | S3互換，ローカル／他クラウドへ差し替え可能 | S3，MinIO，GCS |
| メタデータ | SQLiteから開始，PostgreSQLへ移行可能 | 数日MVPで運用が軽い | 初期からPostgreSQL |
| API契約 | OpenAPI 3.1＋JSON Schema | 言語横断の生成と検証 | Protobufは内部ワーカー分離時に再検討 |

Rust採用のリスクは，チームの学習コストと統計ライブラリの選定です．そのため，MNL数式の単体試験とPython等の参照実装との数値一致試験を重視します．Rustで数日以内の品質確保が難しいと判明した場合は，API契約を変えずにMNLランナーだけを一時的な隔離Pythonサービスへ差し替えます．これは失敗ではなく，契約分離による安全な退避策です．

## 15. リポジトリ構成案

Rustではパッケージ名を`davis-core`，コード内crate名を`davis_core`とする慣例に合わせます．

```text
davis/
  Cargo.toml
  crates/
    davis-contracts/
    davis-core/
    davis-catalog/
    davis-api/
    davis-fmt/
    davis-mnl/
    davis-viz/
    davis-runner/
    davis-cli/
  apps/
    web/
    desktop/
  catalog/
    datasets/
  schemas/
    v1/
  examples/
    mnl-basic/
  packages/
    dataset_cli/       # 移行期間中の既存Python CLI
  legacy/
    models/            # 移行完了後に既存モデルを移す候補
  docs/
    adr/
    davis-platform-concept.md
```

MVP開始時に既存`src/`や`packages/dataset_cli`を移動しません．新基盤が同等機能を持ち，移行試験が通った段階で別PRとして整理します．

## 16. 実装優先順位と数日MVP

### P0: 3〜5日で縦に通す

#### Day 0: 安全確保と決定

- 現在平文保存されているGoogle OAuthクライアントシークレットを失効・再発行する
- データの公開範囲とログイン要否を決める
- MVP用の小さく再配布可能なデータセットを1つ選ぶ
- Rust，TypeScript，R2の開発資格情報を準備する

#### Day 1: 契約と骨格

- Cargo workspaceを作る
- `davis_contracts`へDataset，ArtifactRef，MnlSpec，EstimateResult，Jobを定義する
- JSON SchemaとOpenAPIの生成または固定スキーマを用意する
- `ArtifactStore`のローカル実装とR2実装を作る
- 既存YAMLから新`dataset.yaml`への変換サンプルを1件作る

#### Day 2: MNLの最短経路

- Parquet／CSV読込とlong形式検証を実装する
- 効用式の最小パーサーまたは構造化式を実装する
- MNL尤度，勾配，最適化，共通結果を実装する
- 合成データと参照実装による数値試験を作る
- ローカルCLIで`fit`を実行可能にする

#### Day 3: APIとWeb

- データセット一覧・詳細・署名付きダウンロードURLのAPIを作る
- 推定ジョブの作成・状態・結果APIを作る
- Webにカタログ，MNL設定フォーム，結果表を作る
- 基本的なVega-Liteグラフを1つ表示する

#### Day 4〜5: 品質と公開準備

- 入力エラーと未収束の説明を整える
- E2E試験を1本作る
- 監査用provenanceと成果物ダウンロードを完成させる
- R2上のサンプルデータでステージング動作を確認する
- 利用手順と開発者向け拡張手順を記載する

P0で削ってよいものは高度なデザイン，PCアプリ，任意プラグイン実行，多人数向け権限管理です．削ってはいけないものは入力検証，収束表示，結果契約，ハッシュ，ライセンス表示，秘密情報の分離です．

### P1: MVP後の1〜3週間

- `davis_fmt`の変換レシピとwide／long変換
- 外部モデルランナーのv1プロトコル
- 予測，弾力性，限界効果，可視化
- CLIのRust移植と既存コマンド互換
- PostgreSQLと永続ジョブキュー
- ロバスト標準誤差，重み，パネル対応
- 管理者向けデータ登録・検証・公開コマンド
- 日本語／英語UIとアクセシビリティ

### P2: 1〜3か月

- Tauri PCアプリとオフライン実行
- Nested Logit，Mixed Logitなどの追加モデル
- OCI／WASIモデルプラグイン
- シナリオ比較と実験管理
- 組織，プロジェクト，権限，利用規約同意の管理
- 大規模ワーカー，計算資源制限，利用量管理
- 外部データカタログやGISとの連携

## 17. セキュリティとデータガバナンス

現状の`.dvc/config.local`にはGoogle OAuthクライアントシークレットが平文で存在します．Git管理外であっても，漏えい可能性を考慮して失効・再発行を推奨します．新基盤では次を必須とします．

- 秘密情報は環境変数またはデプロイ先のSecret Managerへ保存する
- Web，CLI，PCアプリへR2の永続資格情報を配布しない
- 非公開ファイルは短時間の署名付きURLで取得する
- 署名付きURLをBearer tokenとして扱い，ログへ完全なURLを出さない
- アップロードファイルのサイズ，形式，拡張子だけでなく内容を検証する
- 元データ，整形データ，推定成果物ごとにアクセス区分と保持期限を持つ
- 個人情報を含むデータについて，ログ，プレビュー，エラーへの値の露出を防ぐ
- ライセンス同意が必要なデータは，同意記録とダウンロード監査を残す
- 外部モデルはサンドボックス化するまで一般利用者に実行させない

## 18. 試験戦略

### 契約試験

- JSON Schemaのvalid／invalid fixture
- Rust型，OpenAPI，TypeScript生成型の整合
- 古いマイナー版の入力を新実装が読めること

### 数値試験

- 手計算可能な小規模2項／多項Logit
- 合成データで真値付近へ回復すること
- 既存Python実装または信頼できる参照実装との尤度・係数比較
- 極端な効用値でNaNやoverflowが発生しないこと
- 共線性，選択肢欠落，完全分離，未収束を正しく警告すること

### 統合試験

- ローカルArtifactStoreとR2互換テスト環境で同じ契約試験を実行
- データ公開からカタログ反映，ダウンロードまで
- アップロード，整形，推定，結果表示，成果物取得まで

### UI試験

- 初学者がサンプルデータから説明なしで完了できるユーザビリティ試験
- キーボード操作，色だけに依存しない警告，日本語の長文表示

## 19. 主要リスクと対策

| リスク | 影響 | 対策 |
| --- | --- | --- |
| 数日で範囲を広げ過ぎる | どの機能も完成しない | 1データ，1モデル，1グラフの縦切りに固定 |
| 共通化を急ぎ過ぎる | 抽象だけが増える | 2つ目の実装が必要になるまで抽象を最小限にする |
| Rust統計実装の誤り | 推定結果への信頼を失う | 参照実装比較，有限差分照合，既知fixture |
| 列名標準化で原情報を失う | 再現不能になる | source／canonical／semanticの3層を保持 |
| YAMLとDBが不一致になる | 説明と実体がずれる | 公開パイプラインだけが索引を生成し，ハッシュで検証 |
| 任意コード実行 | 情報漏えい，サービス停止 | P0では禁止し，後に隔離ランナーを導入 |
| R2固有機能への結合 | 保存先変更が困難 | `ArtifactStore`とS3互換APIを境界にする |
| 初学者向け簡略化で誤解を招く | 不適切な解釈 | 前提，警告，診断を隠さず平易に説明 |

## 20. 先に作るADR

実装開始時に次のArchitecture Decision Recordを短く作成します．

1. ADR-0001: Rustを標準バックエンド言語とするか
2. ADR-0002: 共通表形式をParquet＋Arrowとするか
3. ADR-0003: R2を第一ストレージとし，DVCを管理用途へ限定するか
4. ADR-0004: MNLの規範入力をlong形式とするか
5. ADR-0005: 可視化契約をVega-Lite JSONとするか
6. ADR-0006: MVPの認証方式とデータ公開区分

## 21. 要確認事項

以下は設計上の事実が不足しているため，本書では確定していません．回答後にADRとMVPスコープへ反映します．

1. **利用者と公開範囲**: 一般公開，夏の学校参加者限定，研究室内限定のどれを想定するか．データセットごとに異なるか
2. **MVPの操作範囲**: Webはダウンロードだけか，データアップロードからMNL推定・結果表示までを数日MVPへ含めるか
3. **実行場所**: MNLはサーバー上で実行するか，利用者PC内で実行するか，両方必要か
4. **MVPデータ**: 再配布とWeb処理が可能な代表データセットはどれか．個人情報や契約上の制限はあるか
5. **期待規模**: 典型的な行数，ファイル容量，同時利用者数，許容推定時間はどの程度か
6. **MNL要件**: 代替固有変数，個人固有変数，availability，標本重み，パネル，ロバスト標準誤差のうちP0必須はどれか
7. **運用環境**: Cloudflare中心に寄せるか，既存の大学サーバーや別クラウドを利用するか
8. **開発体制**: Rust，TypeScript，統計実装を担当できる人数と経験はどの程度か
9. **ライセンス**: Davis本体の公開ライセンスと，各データセットの利用条件をどうするか

## 22. 直近の着手順

要確認事項への回答を待つ間も，依存しない次の作業は開始できます．

1. `davis_contracts`の最小スキーマをfixture付きで作る
2. 既存`DatasetConfig`から新`dataset.yaml`への対応表を作る
3. ローカルArtifactStoreを実装する
4. 既存MNLの入出力と数値挙動を回帰fixtureへ固定する
5. long形式の小さな公開可能サンプルを作る
6. OpenAPIの最小APIを起動する
7. その後にR2，認証，Web UIを接続する

## 23. 参考資料

- [Cloudflare R2の仕組み](https://developers.cloudflare.com/r2/how-r2-works/)
- [Cloudflare R2のS3 API](https://developers.cloudflare.com/r2/api/s3/)
- [Cloudflare R2の署名付きURL](https://developers.cloudflare.com/r2/api/s3/presigned-urls/)
- [DVCのS3およびS3互換ストレージ設定](https://doc.dvc.org/user-guide/data-management/remote-storage/amazon-s3)
- [Apache Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html)
- [Polars User Guide](https://docs.pola.rs/user-guide/getting-started/)
- [Vega-Lite Documentation](https://vega.github.io/vega-lite/docs/)
- [Tauri Architecture](https://tauri.app/concept/architecture/)
