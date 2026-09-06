# Davis Component Authoring Acceptance

この文書は，component作成機能が特定の開発者だけに依存していないかを確認する受入手順です．同じcomponent契約を，初学者，YAMLを直接扱う専門家，Davis開発の文脈を知らない外部AIの3経路で利用できれば合格です．

最初に共通の基準componentを生成します．

```console
davis component scaffold ./persona-component \
  --id acceptance/persona-component \
  --kind transform \
  --template python
davis component validate ./persona-component
davis model run ./persona-component/examples/minimal/analysis.yaml
```

合格時には，`summary`の`rows`が3，`sum`が60となり，`output_table`も作られます．これは雛形自体の正解を確かめる基準であり，各personaの課題では壊さないようにします．

## Persona A: 初学者のGUI利用者

この利用者はYAMLを編集しません．Desktopで`examples/minimal/analysis.yaml`を読み込み，次の操作を行います．

1. 入力CSVを別のCSVへ変更します．
2. 「集計する数値列」を列一覧から選びます．
3. 元Planを上書きせず，別名のPlanとして保存します．
4. 推定・処理を実行します．
5. 結果画面で`summary`と`output_table`を確認します．

次をすべて満たせば合格です．

- Manifestに書かれた表示名だけで入力と設定の意味が分かります．
- filesystemの絶対pathや`request.json`を入力する必要がありません．
- 保存したPlanを再読込すると同じ設定が復元されます．
- 実行後，成果物が画面内に表示され，保存先も開けます．

## Persona B: YAMLを直接扱う専門家

この利用者は`component.yaml`とAnalysis Planを直接編集します．`columns.value`を別の数値列へ変更し，入力CSVも差し替えます．必要であれば`component.py`へ平均値等の新しい計算とartifactを追加します．

次をすべて満たせば合格です．

- `configuration.schema`，`presentation.ui`，Analysis Planの`config`が同じ設定構造を指します．
- 新しいartifactを`outputs.artifacts`と`run-result.json`の両方へ宣言しています．
- `davis component validate ./persona-component`が成功します．
- repositoryの場所やcurrent working directoryに依存せず，Planを実行できます．
- 実行条件はcomponent自体ではなく，保存したAnalysis Planに残ります．

## Persona C: Davisの文脈を知らない外部AI

新しい会話のAIへ，[Davis Component Authoring Guide](davis-component-authoring.md)だけをDavis固有仕様として渡します．過去の設計議論，repository内の既存component，Davisのsource codeは渡しません．その上で，次の独立課題を依頼します．

```text
添付したDavis Component Authoring GuideだけをDavis固有仕様の根拠にしてください．
trips.csvを受け取り，generalized_cost = time * time_weight + cost * cost_weight
を各行へ追加するtransform componentを作ってください．

入力列名，time_weight，cost_weightは実行ごとに変更可能にしてください．
出力はgeneralized_costを追加したtext/csvと，行数・平均値を持つapplication/jsonです．
Python標準libraryだけを使ってください．

component.yaml，実program，5行以下のsample CSV，Analysis Plan，test手順を出力してください．
未記載のDavis独自fieldを推測で追加しないでください．
```

次をすべて満たせば合格です．

- 出力されたpackageだけで`davis component validate`が成功します．
- 実programは`--request`でJSONを受け取り，`inputs.<slot>.resolved.path`を読みます．
- 成果物は`output_directory`内へ書き，正しい`run-result.json`を返します．
- 入力，列対応，重み，成果物がManifestとAnalysis Planで矛盾していません．
- sampleの期待値を人間が手計算でき，実行結果と一致します．
- 不明なDavis独自fieldや開発repositoryへの依存がありません．

AIの出力codeは信頼済みとはみなさず，人間が内容を確認した隔離環境で実行します．失敗した場合は，「AIが悪かった」だけで終わらせず，ガイドのどの記述が不足または曖昧だったかを記録し，ガイドかcontract testへ反映します．

## Release判定

3経路の結果を次の表へ記録します．

| 経路 | validate | sample実行 | GUI再編集 | cwd非依存 | 判定 |
| --- | --- | --- | --- | --- | --- |
| 初学者GUI |  |  |  |  |  |
| 専門家YAML |  |  | 対象外 |  |  |
| 外部AI |  |  | 可能なら確認 |  |  |

Python雛形の生成・検証・sample実行はDavisの自動testでも確認します．一方，表示名が初学者に理解できるか，外部AIがガイドだけで新規処理を作れるかは人間を含む受入試験として残します．
