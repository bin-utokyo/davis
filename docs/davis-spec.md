# Davis 仕様書

> 状態: Draft 0.12
>
> 最終更新日: 2026-08-20
>
> 1〜50節は受領したGit-nativeデータ基盤仕様を本文の土台とし，51節以降で交通行動モデル研究プラットフォームとしての拡張を定義します．
>
> 未確定事項は推測で補わず，「要確認」と明示します．

## 1. 概要

### 1.1 プロジェクト名

**Davis**

Davis は，大容量データを Git とオブジェクトストレージを組み合わせて管理するための，Rust 製データバージョン管理・データカタログ基盤です．このデータ基盤を土台として，任意言語の交通行動モデルを実行・比較・再現する研究実行層を接続します．

Git にはデータそのものを格納せず，データセットのメタデータ，manifest，およびバージョン履歴を格納する．大容量ファイルの実体は Cloudflare R2，Amazon S3，MinIO，ローカルストレージ等の外部ストレージに保存する．

Davis は DVC の全面的な再実装を目的としません．Git，オブジェクトストレージ，ハッシュ計算，並列処理等の低レベル機能は成熟した Rust ライブラリを利用し，Davis 自身は「データセットの意味論」「Git 上のメタデータとストレージ上のデータ実体の対応関係」「同期・キャッシュ・カタログ機能」に責任を持ちます．研究実行層はこの責務を置き換えず，版付きデータ参照を通じて上位から利用します．

---

## 2. 設計思想

### 2.1 基本原則

Davis の中核となる考え方は以下である．

```text
Git
= metadata / history の source of truth

Object Storage
= binary content の source of truth

Davis
= Git metadata と Object Storage の bridge
```

Git は以下を管理する．

* データセット定義
* ファイル一覧
* 各ファイルの Content ID
* サイズ
* データセットメタデータ
* Git commit / branch / tag による履歴

外部ストレージは以下を管理する．

* CSV
* Parquet
* GeoPackage
* GeoJSON
* Shapefile
* 画像
* 動画
* モデルファイル
* ZIP
* その他大容量バイナリ

---

## 3. davis-core のスコープ

### 3.1 davis-core が実装するもの

`davis-core`は以下に責任を持つ．Davis全体の研究実行機能は51節以降で定義します．

* Dataset の概念
* Manifest 仕様
* Content Object の識別
* Working Tree と Manifest の対応
* Git revision と dataset version の対応
* Local Cache
* Push / Pull のオーケストレーション
* Checkout
* Status
* Garbage Collection
* データカタログ向けメタデータ
* Storage backend の設定
* 将来的な DVC metadata import/export

### 3.2 davis-core が実装しないもの

以下は既存ライブラリへ委譲する．

* Git object / tree / commit の内部処理
* Git packfile
* S3 API
* HTTP 通信
* retry
* multipart upload
* TLS
* BLAKE3 の実装
* YAML / TOML parser
* CLI argument parser
* async runtime

また，以下については Git 本体を利用し，Davis 独自実装を原則行わない．

* commit
* branch
* tag
* merge
* rebase
* Git remote
* Git authentication

---

# 4. エコシステム

Davis は以下のコンポーネントから構成する．

```text
                    ┌─────────────────┐
                    │    davis-web    │
                    │  Data Catalog   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  davis-server   │
                    │    optional     │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │         davis-core          │
              │                             │
              │ Dataset / Manifest / Cache  │
              │ Sync / Checkout / GC        │
              └──────────┬─────────┬────────┘
                         │         │
                         ▼         ▼
                      Git       Storage
                                R2 / S3
```

上図をデータ基盤の最小構成とします．交通行動モデル研究まで利用する場合は，`davis-runtime`，`davis-model-runner`，モデルコンポーネント，`davis-viz`を上位層として追加します．`davis-core`からこれら上位層への依存は禁止します．

---

## 5. davis-core

### 5.1 役割

`davis-core` は Davis の唯一の中核実装とする．

CLI や Web UI に固有の処理は含めない．

想定 API:

```rust
let repo = DavisRepository::open(".").await?;

repo.add("data/trips.parquet", options).await?;

repo.status().await?;

repo.push().await?;

repo.pull().await?;

repo.checkout().await?;

repo.gc().await?;
```

依存方向は必ず，

```text
davis-cli ──▶ davis-core

davis-web ──▶ davis-server／CatalogIndex

davis-server ──▶ davis-core

catalog-indexer ──▶ davis-core
```

とし，

```text
davis-core ──▶ davis-cli
```

等の逆方向依存は禁止する．

---

# 6. davis-cli

## 6.1 方針

`davis-cli` は `davis-core` の薄いラッパーとする．

CLI にビジネスロジックを実装してはならない．

想定コマンド:

```bash
davis init

davis add data/trips.parquet

davis add datasets/tokyo/

davis status

davis push

davis pull

davis checkout

davis gc
```

将来的に以下も検討する．

```bash
davis dataset list

davis dataset show tokyo-pt

davis dataset edit tokyo-pt

davis remote add

davis remote list

davis import dvc

davis export dvc
```

---

# 7. davis-web

## 7.1 目的

`davis-web` は Git repository に格納された Davis metadata を読み取り，データカタログとして提供する．

想定表示項目:

* Dataset title
* Description
* Creator
* Organization
* License
* Tags
* Spatial extent
* Temporal extent
* Format
* File list
* File size
* Dataset version
* Git history
* Download
* Schema
* Preview
* Map preview

Git repository 自体を **Data Catalog as Code** として利用する．

---

# 8. davis-server

一般公開データまたはローカル利用だけのdeploymentでは，初期バージョンの必須要素としません．行動モデル夏の学校の公式環境は参加者限定データを扱うため，R2をbootstrapするP0-0で，認証と署名付きURLを提供する最小ServerまたはCloudflare Workerを必須とします．P0-AのCLIとP1のWebが同じAPIを利用します．

以下が必要になった段階で導入する．

* private dataset
* authentication
* organization management
* signed URL
* centralized search
* access control
* upload API
* audit log

初期構成では以下を許容する．

```text
GitHub / GitLab
       │
       │ metadata
       ▼
   davis-web

Cloudflare R2
       │
       └── dataset contents
```

---

# 9. 技術スタック

## 9.1 言語

データ基盤，実行ホスト，参照CLI等の主要基盤コンポーネントは **Rust** で実装します．研究者が編集するモデルコンポーネントはPythonを第一経路とし，Rust，R，Julia等も同じプロセス契約から利用できるようにします．

理由:

* 単一バイナリとして配布しやすい
* データ基盤の実行に Python runtime が不要
* CLI との相性がよい
* 大容量ファイル処理に適する
* async I/O が利用可能
* WASM や FFI 等への展開余地がある

---

## 9.2 Rust ライブラリ

基本候補:

```text
Git
└── gix

Storage
└── Apache OpenDAL

Hash
└── blake3

Serialization
├── serde
├── serde_yaml
└── toml

Async
└── tokio

CLI
└── clap

Error handling
├── thiserror
└── anyhow
```

### 9.2.1 Git

Git 操作には原則 `gix` を利用する．

Davis が Git 内部フォーマットを再実装してはならない．

---

## 9.2.2 Storage

Storage abstraction には原則 Apache OpenDAL を利用する．

Davis 自身が S3 client を直接実装しない．

想定 backend:

* Cloudflare R2
* Amazon S3
* MinIO
* Backblaze B2
* Local filesystem
* Azure Blob Storage
* Google Cloud Storage
* WebDAV
* SFTP

Cloudflare R2 は S3-compatible backend として扱い，R2 固有の Davis backend は原則作成しない．

---

# 10. Repository Structure

初期 workspace:

```text
davis/
├── Cargo.toml
├── crates/
│   ├── davis-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── repository.rs
│   │       ├── dataset.rs
│   │       ├── manifest.rs
│   │       ├── object.rs
│   │       ├── cache.rs
│   │       ├── sync.rs
│   │       ├── checkout.rs
│   │       ├── status.rs
│   │       └── gc.rs
│   │
│   └── davis-cli/
│       ├── Cargo.toml
│       └── src/
│           └── main.rs
│
├── apps/
│   └── davis-web/
│
└── docs/
```

初期段階では過度に crate を分割しない．

必要になった場合のみ，

```text
davis-manifest
davis-storage
davis-git
```

等へ分離する．

---

# 11. Repository Layout

Davis repository は通常の Git repository 内に存在する．

```text
project/
├── .git/
├── .gitignore
│
├── .davis/
│   ├── config.toml
│   ├── datasets/
│   │   ├── tokyo-pt.yaml
│   │   └── census.yaml
│   │
│   └── cache/
│       └── objects/
│
├── data/
│   ├── trips.parquet
│   └── zones.gpkg
│
└── src/
```

Git 管理対象:

```text
.davis/config.toml

.davis/datasets/**
```

Git 管理対象外:

```text
.davis/cache/**

大容量データ本体
```

---

# 12. Content Addressable Storage

## 12.1 Object ID

データ実体は content-addressed object として扱う．

初期ハッシュアルゴリズムは BLAKE3 とする．

Object ID:

```text
blake3:<digest>
```

例:

```text
blake3:72dc6e92e32...
```

ハッシュアルゴリズム名を Object ID に含める．

これにより将来的な，

```text
sha256:
blake3:
```

等の共存を可能とする．

---

## 12.2 Storage Key

Storage 上では以下の形式を基本とする．

```text
objects/
└── blake3/
    └── 72/
        └── dc6e92e32...
```

一般式:

```text
objects/{algorithm}/{digest[0..2]}/{digest[2..]}
```

Object ID から Storage Key への変換規則は Davis specification に属する．

Storage backend ごとに異なる規則を持たせてはならない．

---

# 13. Dataset Model

Object と Dataset を明確に分離する．

```text
Dataset
   │
   ├── metadata
   │
   └── files
          │
          ├── ObjectRef
          ├── ObjectRef
          └── ObjectRef
```

Dataset はユーザーが認識する論理単位である．

Object は Storage 上の不変な binary object である．

---

# 14. Manifest

## 14.1 基本形式

Manifest は Git で管理する．

初期形式は YAML とする．

例:

```yaml
version: 1

dataset:
  id: tokyo-person-trip
  title: Tokyo Person Trip Survey
  description: Person-trip survey dataset.

  creators:
    - Tokyo Metropolitan Government

  license: CC-BY-4.0

  tags:
    - mobility
    - person-trip
    - tokyo

  spatial:
    bbox:
      - 139.0
      - 35.2
      - 140.2
      - 36.1

  temporal:
    start: 2018-10-01
    end: 2018-12-31

files:
  - path: trips.parquet
    object:
      oid: blake3:aaaaaaaa
      size: 728193721

  - path: zones.geojson
    object:
      oid: blake3:bbbbbbbb
      size: 2819231
```

---

# 15. Manifest Versioning

Manifest には必ず schema version を含める．

```yaml
version: 1
```

将来的にフォーマット変更が必要になった場合も，既存 manifest を読み込めるようにする．

例:

```rust
enum Manifest {
    V1(ManifestV1),
    V2(ManifestV2),
}
```

---

# 16. Metadata

データカタログへの展開を考慮し，以下の metadata を扱えるようにする．

必須候補:

* id
* title

任意:

* description
* creators
* publisher
* organization
* license
* tags
* homepage
* source
* citation
* spatial extent
* temporal extent

将来的に以下との相互変換を検討する．

* DCAT
* schema.org Dataset
* DataCite
* Dublin Core

内部 manifest を特定規格へ完全依存させない．

---

# 17. Add

## 17.1 File

```bash
davis add data/trips.parquet
```

処理:

```text
file
 ↓
BLAKE3
 ↓
ObjectId
 ↓
local cache
 ↓
Dataset Manifest
 ↓
.gitignore update
```

`davis add` 時点では remote upload を必須としない．

---

## 17.2 Directory

```bash
davis add datasets/tokyo/
```

Directory は一つの Dataset として扱える．

```text
datasets/tokyo/
├── trips.parquet
├── zones.geojson
└── metadata.json
```

から，

```yaml
dataset:
  id: tokyo

files:
  - path: trips.parquet
    ...

  - path: zones.geojson
    ...

  - path: metadata.json
    ...
```

を生成する．

---

# 18. Cache

Local cache:

```text
.davis/cache/
└── objects/
    └── blake3/
        └── ab/
            └── ...
```

Cache の役割:

* remote download の削減
* checkout 高速化
* upload 重複防止
* working tree materialization

Cache 内 object は content immutable とする．

同一 Object ID の content が変更されてはならない．

---

# 19. Push

```bash
davis push
```

現在の Git tree / manifest により参照されている object を remote storage に同期する．

処理:

```text
Manifest
 ↓
ObjectRefs
 ↓
remote existence check
 ↓
missing objects
 ↓
parallel upload
```

すでに remote に存在する Object ID は再アップロードしない．

---

# 20. Pull

```bash
davis pull
```

現在の manifest が参照する object のうち local cache に存在しない object を remote から取得する．

```text
Manifest
 ↓
ObjectRefs
 ↓
local cache check
 ↓
missing
 ↓
remote download
```

---

# 21. Checkout

```bash
git switch experiment

davis checkout
```

Davis 自身は branch を持たない．

Git checkout 後の manifest を読み取り，その Git revision が要求するデータ状態を working tree に再現する．

```text
Git HEAD
 ↓
Manifest
 ↓
Object ID
 ↓
Local Cache
 ├── exists
 │     ↓
 │ materialize
 │
 └── missing
       ↓
      remote
       ↓
      cache
       ↓
    materialize
```

---

# 22. Git Integration

Git が管理する概念:

```text
commit
branch
tag
merge
history
```

Davis が管理する概念:

```text
dataset
object
manifest
cache
sync
checkout
```

そのため初期バージョンでは，

```bash
davis commit
```

を実装しない．

ユーザーは通常通り，

```bash
git add .
git commit -m "Update dataset"
```

を利用する．

---

# 23. Status

```bash
davis status
```

以下を検出する．

* Working Tree のファイル変更
* Manifest と実ファイルの hash mismatch
* Cache に存在しない object
* Remote に存在しない object
* Manifest に存在するが working tree に存在しないファイル
* Manifest に存在しない unmanaged large file

想定表示:

```text
Dataset: tokyo-person-trip

Modified:
  trips.parquet

Missing locally:
  zones.geojson

Not pushed:
  blake3:abc123...

Up to date:
  metadata.json
```

---

# 24. Garbage Collection

```bash
davis gc
```

初期段階では local cache GC を実装する．

将来的に remote GC を追加する．

Remote GC は Git history から reachable Object ID を計算する必要がある．

```text
Git revisions
 ↓
Manifest
 ↓
reachable Object IDs
 ↓
Storage objects
 ↓
unreachable candidates
```

Remote GC は破壊的操作であるため，慎重な仕様とする．

初期実装では dry-run を必須とすることを推奨する．

```bash
davis gc --remote --dry-run
```

---

# 25. Storage Configuration

`.davis/config.toml`

例:

```toml
version = 1

[remote.default]
type = "s3"
bucket = "davis-data"
endpoint = "https://ACCOUNT_ID.r2.cloudflarestorage.com"
region = "auto"
```

Credential は原則 Git 管理される config に直接保存しない．

以下から取得する．

* environment variables
* OS credential store
* AWS-compatible credential chain
* secret manager

---

# 26. Cloudflare R2

Cloudflare R2 は S3-compatible backend として扱う．

例:

```text
type = s3

endpoint =
https://<ACCOUNT_ID>.r2.cloudflarestorage.com

region = auto
```

R2 固有ロジックは必要最小限に留める．

OpenDAL が吸収できる場合，Davis 内部には R2-specific code を持たない．

---

# 27. Transfer

大容量ファイルを対象とするため，以下を考慮する．

* streaming
* multipart upload
* parallel transfer
* retry
* timeout
* integrity verification

これらの低レベル実装は Storage library に任せる．

Davis は，

```text
which object should be uploaded?
```

のみ判断する．

---

# 28. Integrity

Upload / Download 後は必要に応じて Content ID を再計算し，integrity を検証できるようにする．

```bash
davis verify
```

将来的な候補:

```bash
davis verify

davis verify --remote

davis verify tokyo-person-trip
```

---

# 29. Large File Handling

大容量ファイルを一括で memory に読み込んではならない．

Hash 計算は streaming で実施する．

Storage 転送も streaming を基本とする．

---

# 30. Chunking

初期バージョンでは，

```text
1 file = 1 object
```

とする．

例:

```text
trips.parquet
       ↓
blake3:abc
```

Content-defined chunking は初期スコープ外とする．

将来的に，

```text
Large File
   ↓
Chunks
├── object A
├── object B
├── object C
└── object D
```

を導入できる設計余地は残す．

ただし manifest の安定性を損なうような早期導入は避ける．

---

# 31. Concurrency

以下は並列化可能とする．

* hashing
* remote existence checks
* upload
* download

Tokio task を用いる．

ただし同時実行数には制限を設ける．

例:

```text
hash workers     4
upload workers   8
download workers 8
```

値は config で変更可能にする．

---

# 32. Locking

同一 repository に対する，

```text
push
pull
checkout
gc
```

の競合に備え，repository lock を導入する．

例:

```text
.davis/lock
```

Crash 時に永久 lock とならない仕組みを設ける．

---

# 33. Error Model

`davis-core` では型付き error を基本とする．

例:

```rust
pub enum DavisError {
    RepositoryNotFound,

    InvalidManifest,

    ObjectNotFound,

    HashMismatch,

    StorageError,

    GitError,

    IoError,
}
```

CLI では人間向けメッセージへ変換する．

---

# 34. davis-core Public API

初期 API イメージ:

```rust
pub struct DavisRepository {
    // internal
}

impl DavisRepository {
    pub async fn open(
        path: impl AsRef<Path>
    ) -> Result<Self>;

    pub async fn init(
        path: impl AsRef<Path>
    ) -> Result<Self>;

    pub async fn add(
        &self,
        path: impl AsRef<Path>,
        options: AddOptions,
    ) -> Result<DatasetManifest>;

    pub async fn status(
        &self
    ) -> Result<RepositoryStatus>;

    pub async fn push(
        &self
    ) -> Result<SyncReport>;

    pub async fn pull(
        &self
    ) -> Result<SyncReport>;

    pub async fn checkout(
        &self
    ) -> Result<CheckoutReport>;

    pub async fn gc(
        &self,
        options: GcOptions,
    ) -> Result<GcReport>;

    pub async fn datasets(
        &self
    ) -> Result<Vec<Dataset>>;
}
```

---

# 35. API Design Principles

Public API では以下を守る．

* CLI 特有の型を含めない
* `println!` を core で利用しない
* process termination を core で行わない
* environment variables を深い domain layer で直接読まない
* UI message を返さない
* structured result を返す
* side effects を明示する

例:

```rust
pub struct SyncReport {
    pub uploaded: Vec<ObjectId>,
    pub skipped: Vec<ObjectId>,
    pub failed: Vec<SyncFailure>,
}
```

CLI はこれを，

```text
Uploaded: 12 objects
Skipped:  83 objects
Failed:    0 objects
```

と表示する．

---

# 36. davis-web と Core の境界

Core は以下を提供できる．

```rust
repo.datasets()

repo.dataset(id)

repo.dataset_versions(id)

repo.object_metadata(id)
```

Web 固有の以下は core に含めない．

* HTML
* pagination
* HTTP routing
* login session
* UI state
* CSS
* search UI

ブラウザ上の`davis-web`からRustのCoreを直接呼びません．`davis-server`または公開時に生成したCatalogIndexを介します．Core APIはServer，indexer，ローカルclient等が利用します．

---

# 37. Data Catalog Index

Dataset 数が増えた場合，毎回 Git tree 全体を読むのではなく catalog index を生成できるようにする．

例:

```text
Git manifests
     ↓
indexer
     ↓
SQLite / Tantivy / JSON index
     ↓
davis-web
```

ただし index は derived data とし，source of truth は Git metadata とする．

---

# 38. Search

将来的な `davis-web` では以下の検索を想定する．

* full text
* tag
* creator
* organization
* license
* spatial
* temporal
* format

Spatial metadata が存在する場合は地図検索も検討する．

---

# 39. Download

公開 dataset では `davis-web` から直接 object storage へダウンロードさせる構成を許容する．

Private dataset では presigned URL 等を利用する．

```text
Browser
   │
   │ request
   ▼
davis-server
   │
   │ signed URL
   ▼
Browser
   │
   ▼
Object Storage
```

大容量ファイルを application server 経由でproxyしないことを基本とする．

---

# 40. DVC Compatibility

Davis を DVC の fork または完全互換実装とはしない．

ただし migration のため，将来的に以下を検討する．

```bash
davis import dvc

davis export dvc
```

対応候補:

* `.dvc`
* `dvc.yaml`
* `dvc.lock`

初期段階では `.dvc` metadata import を優先する．

DVC cache layout との完全互換は目標としない．

---

# 41. davis-core Non-goals

少なくとも初期バージョンの`davis-core`では以下を実装しません．実験履歴やmetrics可視化はDavis全体から除外するのではなく，51節以降の`davis-runtime`と`davis-viz`へ分離します．

* DVC pipeline DAG
* reproducible pipeline execution
* experiment tracking
* hyperparameter management
* metrics visualization
* ML model registry
* Git implementation
* S3 implementation
* custom HTTP protocol
* custom object database
* content-defined chunking
* distributed locking server
* centralized authentication server

---

# 42. Security

Credential を manifest に含めてはならない．

Secret を以下へ書き込んではならない．

```text
.davis/config.toml
dataset manifest
Git repository
```

対象:

* access key
* secret access key
* API token
* signed URL

Private storage の認証は runtime に解決する．

---

# 43. Logging

`davis-core` では structured tracing を利用可能な設計とする．

推奨:

```text
tracing
tracing-subscriber
```

CLI:

```bash
davis --verbose push
```

等で詳細ログを表示可能とする．

Library 利用時は呼び出し側の subscriber を尊重する．

---

# 44. Testing

## 44.1 Unit Test

対象:

* Object ID
* Storage key
* Manifest serialization
* Manifest validation
* Hashing
* Reachability calculation
* Status comparison

---

## 44.2 Integration Test

Local filesystem backend を利用する．

```text
tempdir
├── git repo
├── davis repo
└── fake remote
```

R2 を必須としない．

---

## 44.3 Backend Compatibility Test

必要に応じて以下を CI / manual test する．

* Cloudflare R2
* AWS S3
* MinIO

---

# 45. Performance Requirements

初期目標:

* TB 級 dataset を扱える設計
* ファイル全体を memory に読み込まない
* upload/download は streaming
* remote object existence check を並列化可能
* 数千〜数万ファイルの manifest に対応可能

大量の小ファイルについては別途 benchmark を行う．

---

# 46. Version Model

Davis は独自の version number を必須としない．

Dataset version は原則 Git revision によって表現する．

```text
Git commit A
    ↓
dataset state A

Git commit B
    ↓
dataset state B
```

Git tag を dataset release として利用できる．

```text
v1.0.0
v1.1.0
```

Davis が Git history を重複管理しない．

---

# 47. Fundamental Invariants

Davis 実装では以下を不変条件とする．

### Object immutability

同一 Object ID に異なる content が存在してはならない．

### Manifest reproducibility

同じ Git revision と同じ remote object set があれば，同じ dataset state を復元できる．

### Metadata in Git

Dataset の意味を決定する metadata は Git repository に存在する．

### Content outside Git

大容量 binary content は Git object database に保存しない．

### Backend independence

Object identity は storage backend に依存しない．

---

# 48. davis-coreの長期機能範囲

本節は，受領した仕様に基づくCoreと周辺機能の到達範囲を示します．記載順を直近の実装順序とはしません．実際には既存DVC資産を活用して読取経路から着手し，CLI，Web，推定器の順に実装します．優先順位と各releaseの完了条件は59節を正とします．

## 機能群1

最小機能:

```text
davis init
davis add
davis status
davis push
davis pull
davis checkout
```

実装:

* Rust workspace
* davis-core
* davis-cli
* BLAKE3
* manifest v1
* local cache
* OpenDAL
* R2 / S3
* local filesystem backend
* Git integration
* `.gitignore` management

---

## 機能群2

```text
davis gc
davis verify
directory dataset
parallel transfer
structured metadata
```

追加:

* multi-file Dataset
* GC
* integrity checking
* spatial metadata
* temporal metadata
* retries / progress reporting

---

## 機能群3

`davis-web`

機能:

* catalog index
* dataset detail
* version history
* download
* metadata search
* map preview
* schema preview

---

## 機能群4

一般公開またはローカルdeploymentでは，必要に応じて`davis-server`を追加します．参加者限定の夏の学校公式deploymentでは，実際の優先順位を定める59節に従い，R2をbootstrapするP0-0で最小ServerまたはWorkerを導入します．

機能候補:

* authentication
* organization
* private datasets
* signed URL
* central catalog
* access control

---

# 49. Architecture Summary

最終的な構造は以下を基本とする．

```text
                        Davis Ecosystem

          davis-cli    davis-server    catalog-indexer
               \           |           /
                \          |          /
                 ┌──────────────────┐
                 │    davis-core    │
                 │ Dataset/Manifest │
                 │ Cache/Sync/GC    │
                 └──────┬─────┬─────┘
                        │     │
                       Git  OpenDAL
                              │
                       R2/S3/Local

          davis-web ──▶ davis-server／CatalogIndex
```

Git integration:

```text
                 Git

       branch / commit / tag
                 │
                 ▼
          Davis Manifest
                 │
           ┌─────┴─────┐
           ▼           ▼
       Object A     Object B
           │           │
           └─────┬─────┘
                 ▼
          Object Storage
```

---

# 50. プロジェクトの位置付け

Davis は，

> DVC を Rust で再実装したもの

とは定義しない．

より適切には，

> **Git-native data versioning and cataloging system backed by object storage**

と位置付ける．

Davis の価値はストレージプロトコルそのものではなく，

```text
Git revision
        +
Dataset metadata
        +
Content-addressed objects
        +
Object storage
```

を一つの統一されたデータ管理モデルとして提供する点にある．

最終的には，

> **Data Catalog as Code**

を実現する基盤として発展させる．

Git repository にデータセットの意味，構造，履歴を保持し，大容量データ本体を任意の object storage に配置することで，CLI，Web catalog，研究データ管理，組織内データ基盤など複数のユースケースを同一の `davis-core` 上で実現する．

---

# 51. 交通行動モデル研究プラットフォームへの拡張

## 51.1 位置付け

1〜50節のデータ基盤仕様をDavisの根幹とします．`davis-core`はデータ管理だけでも単独利用できます．交通行動モデルの整形，実行，結果整理，再現は，Coreを書き換えず上位の`davis-runtime`から利用します．

```text
Git                         Object Storage
metadata / history          binary content
          \                    /
           \                  /
              davis-core
   Dataset / Manifest / Cache / Sync
                    │
              FileSchema
                    │
             davis-runtime
       Format / Run / Provenance
                    │
       MNL／研究者独自モデル
```

Davis全体の目標は，次のとおりです．

> データの発見・取得から，整形，モデル実行，結果整理，再現までの共通作業をDavisが担い，利用者が「どのようなモデルを構築するか」に集中できる基盤を提供します．

初期Webはデータカタログとダウンロードに集中します．整形，推定，可視化はローカル実行を基本としますが，入口をCLIへ固定しません．CLI，GUI，Notebook，ローカルAPI，遠隔APIは同じユースケースを呼ぶ交換可能なクライアントです．

## 51.2 依存方向

```text
davis-web ──────────────▶ davis-server／CatalogIndex

davis-cli ─┐
davis-app ─┼────────────▶ davis-runtime ─▶ davis-model-runner
Notebook ──┘                    │
                               ▼
                           davis-core
```

次の逆方向依存を禁止します．

```text
davis-core ─X─▶ davis-runtime
davis-core ─X─▶ davis-cli
davis-core ─X─▶ davis-web
davis-runtime ─X─▶ 特定モデルの内部実装
```

---

# 52. Metadata と中央契約

## 52.1 DatasetManifest，FileSchema，CatalogIndex

同じ`manifest`という名称で異なる責務を表さないよう，次を区別します．

| 名称 | 原本／派生 | 単位 | 責務 |
| --- | --- | --- | --- |
| `DatasetManifest` | Git上の原本 | 論理データセット | metadata，版，構成ファイル，Object参照 |
| `ObjectId` | Manifest内の不変参照 | 実データObject | content digestとalgorithm |
| `FileSchema` | Git上の原本 | 実データファイル | 列名，型，説明，地域，年，利用条件 |
| `CatalogIndex` | 公開時に生成する派生物 | カタログ全体 | Web検索，facet，ダウンロード参照 |

現行CLIの`dist/manifest.json`は，CLI版，bootstrap package，データ群と`.dvc`ファイル一覧をまとめたリリース索引です．移行中は互換入力として読み，新仕様では`DatasetManifest`と`FileSchema`から`CatalogIndex`を生成します．

metadataの正本はGit上の`DatasetManifest`と`FileSchema`，実データの正本はR2等のObject Storageとします．`CatalogIndex`は正本から生成する派生物とし，直接編集しません．`.dvc`は移行期間中の互換入力として扱います．

新しいR2環境への取込みでは，運営者が用意する実データと対応する`schema.yaml`を入力とします．Davisが実データをstreamingで読みながらBLAKE3を計算し，content-addressed Object，DatasetManifest，CatalogIndexを生成します．実データObjectはR2，`schema.yaml`とDatasetManifestはGitへ保存します．新環境の正本として`.dvc`を作り直す必要はありません．既存`.dvc`は移行元Objectの取得にだけ利用し，将来必要になった場合は互換出力として生成します．

## 52.2 現行directoryとDataset境界

現行の`data/`は，概ね`data/<category>/<dataset>/...`の構造を持つため，この物理階層を移行時にも維持します．例えば`PP/Matsuyama`，`routes/Shibuya-2021`，`network/yokohama`を1 Datasetの候補とします．一方，`PT_data`と`Tohoku_History`は直下に主要Fileを持つため，top-level directory自体を1 Datasetとして扱います．Dataset内の`raw`，`shapefile`，交通手段，年等の下位directoryは，原則としてDataset内の論理的なFile groupingです．

現行Manifestは各DVC管理対象の相対pathを実質的なIDとし，CLIのprefix一致によってdirectory単位の取得を実現しています．新仕様ではpathの深さだけからDataset境界を恒久的に推測せず，`.davis/datasets/`内の`DatasetManifest`から各Dataset rootを明示します．P0移行toolは現行directoryからManifest候補を生成し，例外を運営が確認できるようにします．

Dataset IDは人間が読めるglobalに一意な値とし，初回移行時は`<category>/<dataset>`を候補にします．IDはManifestへ保存し，以後categoryやpathを変更しても自動変更しません．`category`は別のgroup・facetとして保持します．File IDも初回登録時に保存し，pathを変更しても維持します．既存の相対pathはaliasとして残し，現行CLI相当のprefix指定も互換adapterで解決します．

`PT_data`は初期移行では1つのlegacy Datasetとして維持できます．地域，調査年，調査回等のmetadataが整理できた段階で，`pt/<region>-<year>`等のDatasetへ分割し，`pt`配下から探せる論理階層へ移行します．R2上の実データは`objects/<algorithm>/<digest>`へcontent-addressed Objectとして保存し，この論理階層をObject keyへ直接埋め込みません．そのため，後からDatasetを分割しても同じObjectを再uploadせず，DatasetManifestとCatalogIndexの参照だけを組み替えられます．旧`PT_data`は移行期間中のaliasまたはcollectionとして残します．

## 52.3 既存FileSchemaの利用

2026-08-20時点のcurrent revisionには，255個のDVC管理対象と176個の`*.schema.yaml`があります．Webでは実データファイルごとの`<filename>.schema.yaml`を表示・検索します．データセット単位のYAMLだけで列情報を代替しません．件数は固定値として実装へ埋め込まず，対象Git revisionから生成して全件性を試験します．

初期索引は，主に次を扱います．

* `name.ja`，`name.en`
* `description.ja`，`description.en`
* `city.ja`，`city.en`
* `year`
* `license_.ja`，`license_.en`
* `hash_`
* `columns[].name`
* `columns[].type_.name`
* `columns[].description.ja`，`columns[].description.en`

スキーマがないファイルも，現行CLIから取得できる場合はカタログとダウンロード対象から除外しません．

| 状態 | 処理 |
| --- | --- |
| `schema-ready` | ファイル説明と列情報を検索・表示します |
| `schema-missing` | パス，形式，サイズ等を表示し，スキーマ未整備と示します |
| `schema-invalid` | ダウンロードを維持し，検証エラーを運営へ示します |
| `file-missing` | 公開せず，移行エラーとして扱います |

## 52.4 中央契約

言語やクライアントをまたいで固定する契約は，次の6種類に限定します．

1. `DatasetManifest`
2. `ObjectId`
3. `FileSchema`
4. `ModelManifest`
5. `RunRequest`
6. `RunResult`

検索画面の状態，モデル内部の効用関数，尤度，勾配，最適化器，クラス構成は中央契約に含めません．

利用者が記述する`project.yaml`はMVPに設けません．Davisが解決済み入力，整形，モデル，環境，出力を`run.json`へ自動記録します．一括実験の必要性が確認された場合だけ，将来`experiment.yaml`等を任意機能として検討します．

---

# 53. コンポーネントと実装優先順位

| コンポーネント | 責務 | 初期優先度 |
| --- | --- | --- |
| `davis-core` | Manifest・Object・storageの読取，検証，cache，download，公開 | 初回取込みをP0-0，読取経路をP0-A，反復公開をP0-B，高度な更新系をP3 |
| `davis-catalog` | Dataset・FileSchema検証，list・search，CatalogIndex生成 | P0 |
| `davis-cli` | list，info，getと運営者向け公開操作を提供する最初の参照client | P0-A／P0-B |
| `davis-server` | 参加者認証，CatalogIndex配信，download認可，署名付きURL | 最小DownloadGrantをP0-0，Web APIをP1 |
| `davis-web` | 検索，絞り込み，複数選択，download queue | P1 |
| `davis-model-api` | ModelManifest，RunRequest，RunResultのschema | P2 |
| `davis-runtime` | 入力解決，モデル実行，成果物整理，来歴記録 | P2 |
| `davis-model-runner` | モデルprocess起動，log，結果検証 | P2 |
| `davis-mnl` | 標準MNLの参考component | P2 |
| `davis-model-sdk-python` | Pythonモデル向け補助関数と型 | P2 |
| `davis-fmt` | 既知の変換recipeと外部変換process | 互換に必要な最小処理をP2，一般化をP3 |
| `davis-viz` | 共通結果とモデル固有表示の接続 | P3 |
| `davis-app` | 同じuse caseを使うGUI client | P3 |
| `davis-remote-runner` | 隔離されたserver実行 | P4以降 |

`davis-core`をモデル実行の都合で肥大化させません．`davis-runtime`はCoreの版付きデータ参照を利用しますが，ローカルファイルや別組織のストレージもInput Resolverから利用できます．

## 53.1 独立利用と接続利用

各componentは，隣接componentをすべて導入しなくても主要責務を実行できるようにします．

| Component | 単独でできること | 接続時に増えること |
| --- | --- | --- |
| `davis-core` | localまたはremoteのManifestを開き，Objectを検証・取得します | CatalogとRuntimeへ版付きデータ参照を提供します |
| `davis-catalog` | ManifestとFileSchemaからlist・search・index生成を行います | CLIとWebが同じカタログ意味論を利用します |
| `davis-cli` | local catalogからlist・show・downloadを行います | server adapterを使えば公式catalogも利用できます |
| `davis-server` | Webなしでもcatalog APIとdownload認可を提供します | Web，CLI，外部clientの共通backendになります |
| `davis-web` | CatalogIndexまたはcatalog APIだけで検索・downloadできます | 将来Runtime APIへ接続できます |
| `davis-runtime` | local fileだけでもmodelを実行できます | CatalogRefをCoreで解決し，公式データから実行できます |
| model component | fixtureの`request.json`だけで単体試験できます | Runnerから起動され，共通の来歴と成果物管理を利用します |
| `davis-viz` | 保存済みRunResultを読み表示できます | Runtime直後の自動report生成に利用できます |

単独利用のために同じ処理を複製しません．共有するのはdomain contractとuse caseであり，client固有の表示やtransportではありません．

## 53.2 Layerと依存規則

```text
Contracts
DatasetManifest／ObjectId／FileSchema
ModelManifest／RunRequest／RunResult
        │
        ▼
Use cases
ListCatalog／SearchCatalog／ResolveDownload／RunModel
        │
        ▼
Adapters
CLI／HTTP／Pages／Local FS／R2／DVC compatibility
```

次の規則を守ります．

1. Domain contractはCLI引数，HTML，HTTP header，Cloudflare固有型を含めません．
2. Use caseは構造化されたrequestとresultを受け取り，画面出力やprocess終了を行いません．
3. CLI，Web，Server，GUIはadapterであり，domain logicを持ちません．
4. GUIやWebからCLIをshell実行する方式を正式APIにしません．
5. CoreからCatalog，Runtime，clientへの逆依存を禁止します．
6. Runtimeから特定modelの内部実装への依存を禁止します．
7. Component間に循環依存を作りません．

CLIはRust use caseを同一process内で直接呼べます．WebはRust Coreを直接実行せず，`davis-catalog`が生成したCatalogIndexと，`davis-server`のHTTP APIを利用します．この違いがあっても，Dataset ID，File ID，filter，sort，ObjectRef，error codeの意味は共通にします．

`CatalogQuery`，`DownloadSelection`，`DownloadGrant`等はAPI用DTOとしてOpenAPIで版管理しますが，Gitへ保存する中央contractには加えません．複数選択downloadは`DownloadSelection`へFile IDの集合を渡し，Serverが各Objectの権限と存在を検証してから，開始直前に短寿命のURLを発行します．

---

# 54. 公式配布と任意入力元

## 54.1 夏の学校公式環境

行動モデル夏の学校の公式環境では，データの登録・更新・公開を運営に限定し，参加者は検索・閲覧・ダウンロードだけを行えるようにします．

| 役割 | カタログ閲覧 | ダウンロード | 登録・更新・公開 | 署名付きPUT URL |
| --- | --- | --- | --- | --- |
| `participant` | 可 | 可 | 不可 | 発行しません |
| `operator` | 可 | 可 | 可 | 運営経路だけで発行します |

参加者には短寿命DownloadGrantだけを発行し，WorkerがGrantを検証してR2 Objectをstreamします．参加者用Webへupload endpointを公開せず，運営credentialを配布しません．運営は，Object upload，digest検証，ManifestとFileSchemaのGit上のreview・merge，CatalogIndex生成，本番公開の順に公開します．

Cloudflare dashboardへ追加するのは，deploymentとsecret管理を担当する少数の記名管理者だけとします．日常の運営者はCloudflare Account MemberやR2秘密鍵を必要とせず，運営共通codeから発行した期限付きoperator sessionで`davis push`と`davis publish`を実行します．運営共通codeは年度変更や流出時に差し替え，認証revisionの変更によって既存sessionを失効できます．Super Administratorは引継ぎ可能な2名程度に限定し，参加者や通常のデータ利用者をCloudflare Accountへ追加しません．

親credentialはWorker等の信頼環境だけに置き，operator sessionを検証したWorkerが対象Objectのmultipart uploadとCatalog公開を仲介します．長期credentialは日常の運営端末へ配布しません．

この制限は公式deploymentのpolicyです．`davis-core`の`add`や`push`能力を削除しません．別組織が独自repositoryとstorageを運営する場合は，その組織の権限規則で利用できます．

## 54.2 公開後の訂正・公開停止

誤りへ後から気付いた場合も，公開済みObjectを同じkeyで上書きしません．訂正版を新しいrevisionとして公開し，旧revisionへ`superseded`，`withdrawn`または`revoked`の状態，理由，判明日時，代替revisionを記録します．通常の誤りでは旧Objectを再現性のため保持しますが，Catalogの既定検索と通常の`get`から除外し，明示的に旧revisionを参照した場合は警告します．

個人情報，契約違反等により保持自体が許されない場合は，緊急失効として新規署名付きURLの発行を停止し，該当ObjectをR2から削除します．必要に応じてGit履歴内の機微metadataも別途purgeします．すでに利用者が取得したcopyは技術的に回収できないため，運営上の連絡と利用停止依頼が必要です．訂正履歴と緊急操作はaudit logへ残します．

## 54.3 Runtimeの入力元

`davis-runtime`は公式カタログへの接続を必須としません．次の入力元を同じ検証済みローカル参照へ正規化します．

| `kind` | 用途 | 導入時期 |
| --- | --- | --- |
| `catalog` | 公式または別組織のDavis repository | Runtime導入時のP2 |
| `local` | PC上のファイルまたはディレクトリ | Runtime導入時のP2 |
| `https` | 組織内外のHTTPS download先 | P3 |
| `s3` | R2，S3，MinIO等 | P3 |
| `custom` | 組織固有resolver | P4以降 |

研究者や行政職員がローカルデータや庁内サーバーを利用しても，Davis公式環境へのuploadは発生しません．明示的な公開操作を行わない限り，Runtimeは入力データを外部送信しません．

access key，token，署名付きURL等は`RunRequest`，`run.json`，モデルprocessへ保存・転送しません．Runtimeが環境変数，OS credential store，またはcredential profileから実行時に解決します．共有用の実行記録からは，ローカル絶対パス等の機微情報を除去または相対化します．

---

# 55. モデルコンポーネントAPI

## 55.1 Processとファイルの境界

Python，Rust，R，Julia等を同じDavisから起動できるよう，関数ABIや継承classではなく，processとJSON・Parquetファイルを境界にします．

```text
davis-model-runner
  ├── 入力元をローカル参照へ解決します
  ├── model向けrequest.jsonを作成します
  ├── model processを起動します
  └── output/run-result.jsonを検証します

model component
  ├── request.jsonを読みます
  ├── 任意の方法で推定します
  ├── 可能な標準成果物を出力します
  └── 独自成果物をextensionsへ出力します
```

## 55.2 ModelManifest

```yaml
api_version: davis.model/v1alpha1
id: example-lab/scale-mnl
name: Scale-adjusted MNL
version: 0.1.0

runtime:
  kind: python
  command: ["uv", "run", "python", "-m", "scale_mnl"]
  lockfile: uv.lock

operations: [validate, estimate, predict]
inputs:
  - name: choice_data
    media_types: [application/vnd.apache.parquet]
    required: true
config_schema: schemas/config.schema.json
outputs:
  standard: [parameters, covariance, metrics, predictions]
  extensions: [estimated_scales]
```

MVPは`python`と`native`から始め，`wasm`と`container`を後から追加します．

## 55.3 RunRequest

```json
{
  "api_version": "davis.run/v1alpha1",
  "run_id": "run_01...",
  "operation": "estimate",
  "component": {
    "id": "example-lab/scale-mnl",
    "version": "0.1.0",
    "source_revision": "<git-commit>"
  },
  "inputs": {
    "choice_data": {
      "source": {
        "kind": "catalog",
        "dataset_id": "tokyo-person-trip",
        "revision": "<git-commit>",
        "file_id": "trips.parquet"
      },
      "resolved": {
        "path": "/resolved/input/choice-data.parquet",
        "object_id": "blake3:<digest>",
        "size": 728193721
      }
    }
  },
  "config": {
    "path": "/resolved/input/model.yaml",
    "sha256": "<hash>"
  },
  "output_directory": "/resolved/output"
}
```

クライアントは主に`source`を指定し，Runtimeが`resolved`を生成します．Davisの`run.json`には秘密情報を除いた両方を記録します．モデル向け`request.json`には原則として`resolved`だけを渡します．

## 55.4 RunResult

```json
{
  "api_version": "davis.result/v1alpha1",
  "run_id": "run_01...",
  "status": "succeeded",
  "artifacts": {
    "parameters": "parameters.parquet",
    "covariance": "covariance.parquet",
    "metrics": "metrics.json",
    "predictions": "predictions.parquet"
  },
  "extensions": {
    "example-lab/estimated-scales": "extensions/scales.parquet"
  }
}
```

すべてのモデルへ同じ成果物を強制しません．状態，来歴，成果物参照だけを必須とし，係数表，予測値等は出力可能なモデルの任意標準成果物とします．モデル固有結果は名前空間付き`extensions`へ保存します．

---

# 56. 標準MNLとモデル拡張

`davis-mnl`はDavisに固定された唯一のモデルではありません．モデルコンポーネントAPIの参考実装であり，複製・変更して新しいモデルを作るためのひな型です．利用者は，データ取得，cache，入出力整理，hash，実行記録をモデルごとに書き直さず，主にモデルコードと設定を変更します．

標準MNLの推奨入力はlong形式のParquetとし，`case_id`，`alternative_id`，`chosen`，`available`に相当する列を設定で対応付けます．ただし，現行の`los.csv`と`trip.csv`を初期互換入力として維持するかは要確認です．他モデルにはlong形式を強制せず，複数表，network，GeoJSON，行列等を追加できます．

入力contractは，共通部分とmodel固有部分を分けます．Runtimeが共通に扱うのは，入力slot名，media type，File参照，digest等です．標準MNLは`case_id`，`alternative_id`，`chosen`，`available`という意味上のroleを要求しますが，実データの列名そのものは固定せず設定から対応付けます．説明変数，weight，panel ID，network等の追加要件は各ModelManifestと`config_schema`が宣言します．これにより，標準MNLは共通の入力検証を利用しながら，他modelへlong形式や同一列構成を強制しません．

モデル内部の効用関数，確率，尤度，gradient，optimizer，parameter共有方法は共通classへ固定しません．標準MNLを少し変更したモデルも，独立したModelManifestとprocessとして登録し，同じRunRequestから実行・比較できるようにします．

基盤はRustを第一候補とし，研究モデルはPythonを第一経路とします．Python環境はcomponent単位の`uv.lock`で隔離し，Python executable，lockfile hash，package版を`run.json`へ記録します．Rustだけへ統一することも，Davis全体をPythonへ統一することも目標にしません．

---

# 57. davis-fmt と davis-viz

## 57.1 davis-fmt

`davis-fmt`は全データを万能形式へ変換する機能ではありません．元データから，特定モデルが要求する入力を再現可能に作ります．

1. 列名変更，型変換，値変換，join，wide・long変換は宣言的recipeで記述します．
2. recipeで表現しにくい処理は，Python，Rust等の独立した版付き変換componentとして実装します．

任意codeをYAMLへ埋め込みません．FileSchemaに存在する列と型は検証しますが，列名だけから意味を推測して選択肢や効用へ自動割当てしません．

## 57.2 davis-viz

可視化は次の3層へ分けます．

1. すべてのRunResultに対する状態，来歴，log，成果物一覧
2. parameters，metrics，predictions等の任意標準成果物
3. extensionsを使うモデル固有表示

`parameters.parquet`は`name`と`estimate`を必須列とし，`std_error`，`statistic`，`p_value`，`lower`，`upper`を任意列とします．対応モデルでは係数表，信頼区間図，diagnosticsを共通表示します．係数を持たないモデルには強制しません．

共通HTML・JSON・CSVと，Vega-Lite specificationを第一経路とします．Matplotlib等はモデル固有の任意成果物として利用できます．

---

# 58. Web，CLI，GUI

## 58.1 初期Web

初期Webは，実データファイルごとのFileSchemaを用いたダウンロードカタログに限定します．

* 日本語・英語の名称・説明の検索
* 地域，年，形式，license，schema状態による絞り込み
* 列名，列説明，列型，複数列の包含条件による検索
* Datasetとファイル詳細
* Raw YAML表示
* 短寿命DownloadGrantを検証するWorker経由のdownload

検索のたびに全YAMLを走査せず，公開時に`files.json`，`columns.json`，`facets.json`等を生成します．初期はCloudflare Pagesで静的索引を検索し，規模が増えた場合は同じ応答形式のAPIへ交換します．

## 58.2 クライアントの同列性

CLIは最初の参照クライアントですが，中核APIではありません．

```text
CLI       ─┐
GUI       ─┼─▶ Run use case ─▶ RunRequest／RunResult
Notebook  ─┤
Remote API─┘
```

すべてのクライアントが同じ`run.json`，成果物，実行履歴を利用します．GUIがCLIをshell実行することを正式な接続方式にはしません．

## 58.3 GUIプロトタイプから取り入れる要素

`feature/davis-gui`の固定されたCar・Rail・Bus・Walk editorやmock推定は正式仕様にしません．次の概念はCoreまたは共通成果物へ取り入れます．

* UIとservice adapterの分離
* 選択中データのFileSchemaに基づく列候補と型検証
* データ変更時のモデル設定再検証
* 自動生成された実行履歴と比較
* Table，Coefficients，Diagnosticsの共通表示
* Draft，Modified，Estimated，Saved等の画面状態

## 58.4 Form編集とcode編集

標準MNLのGUIは，まず次の線形効用をFormから作成できるようにします．列候補はFileSchemaから選択し，parameter名，共有関係，初期値，固定・推定，説明変数を編集します．

```text
V_ni = ASC_i + Σ beta_k x_nik
```

Formの構造化設定は標準MNL component固有の`config_schema`に従い，Davis全体の中央contractには加えません．Formと生成codeを同時に別々の正本として管理せず，Form互換modeでは構造化設定を正本としてcodeを生成します．

利用者は画面切替から生成codeを確認・編集できます．codeが線形和，parameter，列参照等の明示した対応構文だけで表現されている間は，構文検査を通してFormへ戻せます．数学的な同値性は推測しません．非線形効用，独自関数，独自尤度，独自class等の非対応構文を使う場合は確認後に，そのmodel revisionを高度な`code` modeへ一方向に切り替えます．`code` modeではFormを編集不可にし，移行直前の構造化設定を参照用に保持します．Formへ戻したい場合は，元revisionから新しいForm互換modelを作成します．

GUI編集とcode編集のどちらで作成したmodelも，同じModelManifest，RunRequest，RunResultを使って実行します．したがって，GUIが表現できないmodelでも，データ解決，実行記録，成果物管理，比較機能は失いません．

## 58.5 インストールと更新

Webだけを使う参加者は何もインストールしません．CLI利用者には，Windows，macOS，Linux向けの小さな`davis`基本binaryをGitHub Releasesから配布し，GitHub URLを使うinstall scriptも用意します．基本binaryは`login`，`list`，`info`，`get`，cache，検証，更新確認を含みますが，Python，MNL，GUI，他modelを同梱しません．Runtime，model component，GUIは必要になった利用者だけが後から追加します．

```text
davis component install runtime
davis component install mnl
```

基本binaryとcomponentは独立してSemVerで版管理します．CLIは対話端末で1日1回以下の頻度でGitHub Releasesの最新版を確認し，新版がある場合だけ変更概要と`davis update`を案内します．利用者の確認なしに実行中のbinaryを自動置換せず，非対話実行とoffline利用では通知を抑止できます．更新時はrelease artifactのchecksumと署名を検証します．Catalog protocolと互換である限り旧CLIを直ちに使用不能にはしません．

---

# 59. 実装順序

## 59.1 基本方針

実装順序は次の段階を最優先とします．

```text
P0-0  実データとschema.yamlを取込み，R2をbootstrap
        ↓
P0-A  CLIでlist・詳細確認・get
        ↓
P0-B  運営者CLIで差分確認・R2公開・検証
        ↓
P1  Webでschema検索・複数選択・download
        ↓
P2  localまたはcatalog入力からモデル推定
        ↓
P3  その他の拡張
```

各段階は，後続componentが未実装でも単独でreleaseできる状態を完了条件とします．同時に，後続段階が既存処理を再実装せず接続できるcontractを残します．

## 59.1.1 実装状況 (2026-08-21監査)

本節以降の「実装対象」と「完了条件」は目標仕様です．記載されているcommandや構成要素がすべて実装済みであることを意味しません．実repositoryとreleaseを照合した現在の状況は次のとおりです．

| 段階 | 状況 | 実装済み | 主な未実装・設計差分 |
| --- | --- | --- | --- |
| P0-0 | 完了 | BLAKE3 Object，DatasetManifest，R2，参加者認証，DownloadGrant，全255 FileのCatalog生成 | 旧DVC remoteの停止判断は運用事項です |
| P0-A | 完了 | `list`，`info`，`get`，`pull`，File・directory単位取得，取得前license表示，schema・日英PDF取得，3 OS向けrelease | loginはbrowser起動ではなくterminal入力，sessionはOS credential storeではなく権限を限定したuser設定fileへ保存します |
| P0-B | 一部完了 | `verify`，差分Objectだけの`push`，review済み`main`限定の`publish`，運営session | 公開revisionとの差分をまとめる`status`と`verify --remote`は未実装です |
| P1 | ほぼ完了 | 日英Web，schema検索・filter，複数選択，利用条件確認，200並列認証test，認証付きdownload | D1は使わず署名済みstateless sessionを採用し，R2署名URLの直接返却ではなく短寿命DownloadGrantをWorkerが検証してstreamします |
| P2 | 未着手 | なし | Runtime，Model API，MNL，format adapter，RunResultは未実装です |
| P3 | 未着手 | 低頻度のCLI更新通知だけを先行実装 | GUI，汎用fmt・viz，GC，履歴通知等は未実装です |

この表を実装状況の正とし，目標仕様との差分が解消された時点で同時に更新します．

## 59.2 P0-0: R2 bootstrap取込み

最初の利用者向けreleaseはCLIですが，その前提として新しいR2取得経路を一度成立させます．現行DVC remoteから必要な実データを復元し，実データと`schema.yaml`をDavisへ入力します．DavisはBLAKE3，Object，DatasetManifest，CatalogIndexを生成し，全ObjectをR2へstreaming uploadします．この段階でDVC repositoryを作り直しません．

同時に，共通招待codeをCLI用sessionへ交換し，短寿命GET URLを返す最小Workerを用意します．P0-0は移行用bootstrapでもよく，反復利用しやすい運営者UXと差分同期はP0-Bで完成させます．

### 完了条件

1. 対象Git revisionの全DVC管理対象を実データとして復元できます．
2. 全実データへBLAKE3 Object IDを付与し，R2でsizeとdigestを検証できます．
3. 全FileSchemaとschema未整備FileからCatalogIndexを生成できます．
4. 最小Worker経由で認証済みのtest clientがR2 Objectを取得できます．
5. 旧DVC remoteを停止しても利用できる新経路のcopyを確保します．

## 59.3 P0-A: CLIによるカタログ閲覧とget

最初のreleaseでは，1〜50節のCore仕様全体を完成させません．P0-0で構築したR2経路を第一経路とし，既存DVC資産を読む互換adapterをfallbackとして残して，読取専用の縦断経路を最短で安定させます．

### 実装対象

1. `davis-core`のManifest読取，Object参照解決，hash・size検証，cache，download
2. 現行`manifest.json`，`.dvc`，`*.schema.yaml`の互換adapter
3. 現行directoryからDatasetManifest候補を生成・確認する移行tool
4. `davis-catalog`のDataset・File一覧，詳細，基本filter，CatalogIndex生成
5. 薄い`davis-cli`
6. Windows，macOS，Linux向けbinaryと全件互換test

### 初期command

```text
davis login <catalog-url>
davis list
davis info <dataset-id>
davis get <dataset-id>
davis get <dataset-id> --file <file-id>...
davis get <dataset-id> -o <directory>
davis pull <dataset-id>
```

`get`は初回取得とFile・directory単位の選択取得，`pull`はDataset全体の初回取得と現在のManifestへの同期取得に使用します．どちらも既定ではcurrent directoryを出力rootとし，その下にDataset内の相対pathを保って配置します．`-o`または`--out`で出力rootを変更できます．`pull`は既存Fileをremoteの内容で更新するため，local編集が残っていない状態で使用します．

`davis get`は，対象Objectと期待digestを含む内部Download Planを作り，storage adapterが取得します．Command自身にManifest解析，remote選択，copy処理を書きません．

### 完了条件

1. 現行CLIで取得できる全Datasetと関連文書を取得できます．
2. 対象Git revisionの全DVC管理対象が一覧とdownload対象から欠落しません．current revisionでは255個です．
3. 対象Git revisionの全FileSchemaを詳細表示に利用できます．current revisionでは176個です．
4. schema未整備のファイルも`schema-missing`として取得できます．
5. CLI以外から同じuse caseを呼ぶunit testを用意します．
6. Web，Runtime，MNLを導入しなくてもCLIだけで動作します．
7. Windows，macOS，Linuxで同じDatasetを一覧・確認・取得できます．

機能互換性はcommand名や内部構造の一致ではなく，現行CLIで可能な一覧，詳細確認，File・directory単位の取得，関連文書の取得が欠落しないことで判定します．

## 59.4 P0-B: 運営者によるR2公開

P1のWeb公開に先立ち，運営者がDavisだけでlocalの変更を確認し，R2へ不足Objectを安全に反映できる最小更新経路を実装します．

```text
davis status
davis push --dry-run
davis push
davis publish
davis verify --remote
```

現行実装では，participantは`davis login`のterminal promptへ共通招待codeを入力し，CLI用session tokenを権限を限定したuser設定fileへ保存します．`davis get`と`davis pull`はP0-0のWorkerから対象Objectごとの短寿命DownloadGrantを取得します．operatorのR2 upload credentialとは別系統にし，participant sessionからPUT，DELETE，Object一覧を許可しません．この認証・DownloadGrant APIをP1のWebでも再利用します．browser起動loginとOS credential storeは将来差し替え可能なadapter候補です．

`status`はlocalのManifest・FileSchema・実データと公開中revisionとの差分を表示します．`push`はcontent-addressed Objectの存在を確認し，不足Objectだけをuploadします．既存Objectの上書き，remote Objectの自動削除，GC，競合merge，Catalog公開は行いません．`publish`はreview済みの最新`main`だけからCatalogを公開します．公開順序は次のとおりです．

```text
Manifest候補と差分を作成
        ↓
不足ObjectをR2へupload
        ↓
sizeとdigestを検証
        ↓
Git上のManifestとFileSchemaをPull Requestでreview・merge
        ↓
最新mainからdavis publish
        ↓
R2 Objectの網羅性を検証してCatalogIndexを公開
```

途中で失敗した場合は新revisionをCatalogへ公開しません．R2 credentialと公開操作はoperatorだけが利用でき，participant向けCLIやWebへ渡しません．

`init`，`add`の一般化，`checkout`，remote削除，GC等の高度な更新系は，この段階の必須条件にせずP3で追加します．

## 59.5 P1: Webによる検索とdownload

P0のDataset・File・Objectの意味をそのまま利用し，検索・認証・複数選択を追加します．Webのために別のcatalog生成処理やdownload処理を作りません．

### 実装対象

1. `davis-catalog`によるfile，column，facetの静的index生成
2. Pages上の検索，filter，Dataset・File詳細，Raw YAML表示
3. File単位とDataset単位のcheckbox，選択drawer，合計size表示
4. 選択したFile ID集合を受け取るDownloadSelection API
5. 年度単位の共通招待code，session cookie，operator・participant権限
6. 短寿命DownloadGrantとdownload queue
7. 署名済みstateless session，Worker／Pages Functions，R2の接続

### 独立性

`davis-web`はCatalogIndexとHTTP APIだけで動き，RuntimeやMNLを要求しません．`davis-server`はWebなしでもAPI testとCLI用remote adapterから利用できます．CLIはP1後もlocal・互換経路で動作し，Serverを必須にしません．

### 完了条件

1. FileSchemaの名称，説明，地域，年，形式，列名，列説明，列型を検索できます．
2. 複数FileまたはDataset全体を選択し，queueから順次取得できます．
3. 参加者は閲覧・downloadだけを行い，upload・更新・PUT URL発行を行えません．
4. 100〜200人規模を想定した認証・download負荷testを通します．
5. CLIとWebでDataset ID，File ID，size，digest，download対象が一致します．

Catalog metadataと検索indexは公開情報としてPagesから配信できます．実データのdownloadだけを共通招待codeとsession cookieで保護します．招待codeはclient側へ埋め込まずServer側で検証し，年度更新または流出時に差し替えられるようにします．code差替え時には旧codeで発行したsessionも失効できるよう，sessionを認証revisionへ紐付けます．

sessionの初期有効期間は180日を上限とし，年度切替またはcode差替え時には残存期間にかかわらず失効できるようにします．

Webは選択したFileを個別に順次downloadし，選択内容に対応する`davis get` commandのcopyも提供します．browserは任意のdirectory階層へ自動配置できないため，個別選択時の保存先と階層はbrowser設定に従います．Dataset全体については，公開時に生成した任意のZIP bundleがある場合に限り，階層を保った一括downloadを提供します．大量File，大容量Dataset，任意選択での階層維持にはCLIを案内します．ZIP bundleは派生物であり，実データの正本にはしません．

現在の全FileSchemaには，行動モデル夏の学校に利用を限定する`license_`が記載されています．WebとCLIは取得前に対象Fileのlicenseを表示し，CatalogIndex生成時に欠落を検証します．

## 59.6 P2: 推定器

カタログ機能が安定した後，研究実行層を追加します．RuntimeはWebへ埋め込まず，まず利用者PC内で実行します．

### 実装対象

1. `davis-model-api`のModelManifest，RunRequest，RunResult
2. `davis-runtime`のInput Resolver，実行directory，来歴記録
3. `davis-model-runner`のprocess起動，log，終了状態，成果物検証
4. 現行Python MNLを接続する`davis-mnl`
5. component単位の`uv.lock`とPython環境管理
6. 現行入力を推奨入力へ接続する必要最小限のformat adapter
7. parameters・metrics等の基本CSV・JSON出力

### 独立性

Runtimeはlocal fileだけで実行でき，公式catalogを必須にしません．Catalogと接続する場合は`catalog` InputRefをCoreでlocal pathへ解決します．MNL componentはfixtureの`request.json`から単体実行・testできます．CLI，GUI，Notebookは同じRun use caseを呼びます．

### 完了条件

1. catalogとlocalの両入力を同じRun use caseから利用できます．
2. 既存MNLを独立componentとして実行できます．
3. 標準MNLを変更したcomponentをDavis本体のforkなしに実行できます．
4. 入力digest，model revision，環境，RunRequest，RunResult，成果物を`run.json`へ記録できます．
5. CatalogとWebがなくてもlocal推定を実行できます．

最初の実データ例は`Tohoku_History`の居住地選択データとし，`df_individual.csv`の選択結果と`df_ex_var.csv`の選択肢別説明変数から，小規模な回帰test用subsetを作ります．これは推定器の最初の検証対象であり，Catalogとdownloadの全Dataset対応を限定しません．

標準MNLの初版は，線形効用，alternative-specific constant，共通係数・選択肢固有係数，固定parameter，availability，最尤推定，基本的な標準誤差と適合度指標までを対象にします．weight，panel向け分散推定，robust standard errorは後続拡張とします．

## 59.7 P3以降

P0〜P2が安定した後，次を優先度と需要に応じて追加します．

* `davis-core`の`init`，`add`一般化，`checkout`，remote削除，local・remote GC
* `davis-fmt`の一般的なrecipe・外部変換component
* `davis-viz`の係数表，係数図，diagnostics，共通HTML
* GUI，Notebook SDK，実行履歴比較
* authenticated HTTPS，S3，組織固有Input Resolver
* Nested Logit，Mixed Logit，Recursive Logit等の参考component
* OCI，WASI，remote runner
* remote GC，組織・権限，model registry
* 安定したFile IDとObject IDを比較し，過去に取得したFileの更新をWeb上で知らせる任意機能

更新通知は初期要件としません．共通招待codeだけを使う場合，取得履歴は匿名IDを保存した端末・browser単位とします．端末をまたぐ利用者単位の履歴が必要になった場合に限り，個別code等の識別方法を追加します．

---

# 60. 確認済み事項と要確認事項

## 60.1 確認済みの設計事項

この一覧は合意済みの方針であり，すべての実装完了を表すchecklistではありません．現在の実装状況は59.1.1節を正とします．

1. 1〜50節のGit-nativeデータ基盤仕様をDavisの土台とします．
2. 現在の`data/`にある全データをカタログ対象にします．
3. 現行CLIで取得可能な全データのdownload互換性をMVP条件にします．
4. 初期WebはFileSchema検索とdownloadへ限定します．
5. 公式環境のupload・更新・公開は運営に限定します．
6. 参加者は公式データを閲覧・downloadできます．
7. Runtimeはlocal dataや独自serverも入力にできます．
8. 外部入力は明示的な公開操作なしに公式storageへuploadしません．
9. 推定はlocal実行を基本とし，将来server実行を追加可能にします．
10. MNLは固定機能ではなく，変更modelを作る参考componentです．
11. CLIを中核APIにせず，GUI，Notebook，remote APIと同列にします．
12. 利用者が記述する`project.yaml`はMVPに設けません．
13. 実装順序は，R2 bootstrap，CLIのlist・info・get，運営者CLIの反復公開，Webの検索・複数選択download，local推定器，その他の拡張とします．
14. 各componentは単独利用を可能にし，中央contractとuse caseを通じて接続します．
15. CLI，Web，GUIでdomain logicを複製せず，それぞれを共通use caseのadapterとして実装します．
16. 現行の`data/<category>/<dataset>/...`を基本的に維持し，DatasetManifestでDataset境界と安定IDを明示します．
17. metadataは公開可能とし，初期Webでは実データのdownloadだけを共通招待codeで保護します．
18. 共通招待codeは年度ごとの更新と流出時の差替えを可能にし，必要に応じて既存sessionも失効させます．
19. CLIの互換性は利用可能な機能で判定し，選択取得の`get`と同期取得の`pull`を提供します．どちらも初回取得に使用できます．
20. P0のCLIはWindows，macOS，Linuxを対象とし，既定の取得先をcurrent directoryとします．
21. 標準MNLは共通の意味上の列roleとmodel固有の追加要件を組み合わせ，他modelへ同じ入力形式を強制しません．
22. GUIは線形効用をFormと対応構文のcodeから編集でき，非対応構文を使う場合は高度なcode modeへ一方向に移行します．任意codeをFormへ逆変換しません．
23. Model出力は状態，来歴，成果物参照だけを共通必須とし，係数等は任意標準成果物とします．
24. 新R2環境は実データと`schema.yaml`を取込入力とし，DavisがBLAKE3，DatasetManifest，CatalogIndexを生成します．`.dvc`は移行時の互換入力に限定します．
25. P0-0で共通招待codeによるCLI login，DownloadGrant API，認証済みObject streamを提供する最小Workerを導入します．
26. 誤った公開は新revisionで訂正します．旧revisionの状態管理と緊急削除のaudit機能は未実装であり，P3の保持・削除policy実装時に追加します．
27. `davis`基本binaryはGitHub Releasesとinstall scriptから配布し，Runtime，model，GUIを必要時に追加します．
28. CLIは利用者へ確認して更新する`davis update`と，新版の低頻度な確認通知を提供します．
29. Webの個別File downloadではdirectory階層を保証せず，任意のDataset ZIPまたはCLIで階層を維持します．
30. 共通招待codeのsessionは初期値を最大180日とし，年度切替とcode差替えで強制失効できます．
31. 現在の全FileSchemaにある夏の学校限定`license_`をWebとCLIの取得前に表示します．
32. 標準MNLの最初の実データ例は`Tohoku_History`から作る小規模subsetとします．
33. 標準MNL初版は線形効用，ASC，共通・選択肢固有係数，固定parameter，availability，最尤推定，基本標準誤差・適合度指標を対象とします．
34. Cloudflare Account Memberはdeploymentとsecret管理を担う少数の記名管理者に限定し，日常のoperatorは運営共通codeから発行した期限付きsessionを使用します．

## 60.2 要確認事項

1. 各Datasetの確定ID，旧path alias，例外的なDataset境界の対応表
2. 将来複数のaccess区分を設ける場合，DatasetManifestとFileSchemaのどちらで管理するか
3. 現行`los.csv`と`trip.csv`向け互換adapterをいつ追加するか
4. 最初の変更model例として何を採用するか
5. Python SDKをNumPy・SciPy中心にするか，JAX等を採用するか
6. 既存FileSchemaへ将来の検索fieldをどこまで追加するか
7. Webで一度に個別downloadするFile数・合計sizeの上限
8. 同時利用者数，download量，R2費用上限
9. Davis本体とmodel componentのlicense
