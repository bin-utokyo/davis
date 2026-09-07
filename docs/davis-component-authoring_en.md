# Davis Component Authoring Guide

[日本語](davis-component-authoring.md)

This guide explains how to make your own program executable from Davis Desktop and the Davis CLI. A component may estimate a statistical model, join CSV files, calculate explanatory variables, or visualize results. It may be written in Python, R, Julia, Node.js, Java, Rust, C++, or any other language that can obey the process contract below.

In Davis, a **component** is an ordinary program packaged with a description of its inputs, configurable settings, and outputs. That description is `component.yaml`.

Beginners should start with “Key terms” and “Creating your first component.” Experienced authors and AI assistants may use the sections from “Manifest” onward as a specification.

## Key terms

| Term | Meaning | Example |
| --- | --- | --- |
| component | A complete calculation packaged for Davis | MNL estimation or a CSV join |
| Manifest | The component description named `component.yaml` | Declares a CSV input and a parameters output |
| input | A file passed to the program | `persons.csv` |
| config | A setting selected for each run | ID column, variables, or iteration count |
| artifact | An output file created by the program | Coefficient CSV or metrics JSON |
| runtime command | The command that starts the program | `python -m my_model` |
| Analysis Plan | YAML that records one run | Input files, selected columns, and settings |
| schema | Rules for allowed config values | `max_iterations` is an integer of at least 1 |
| presentation | Optional hints for editing config and showing results | Display a result CSV as a table |

```text
component.yaml  -- what the program accepts and returns
program         -- the calculation itself
analysis.yaml   -- files and settings selected for this run
       |
       `-- Davis runs it and stores results and provenance in davis-runs
```

## Basic package structure

A minimal component consists of a Manifest and a program. `component.py` is only an example; another language or a native executable is also valid.

```text
component/
├── component.yaml
└── component.py
```

Davis resolves input files, starts the program, stores logs and results, and verifies that declared outputs were created. The program remains responsible for the domain-specific calculation.

## Creating your first component

Generate a small executable Python component:

```console
davis component scaffold ./my-component \
  --id example/my-component \
  --kind transform \
  --template python
```

Before changing it, verify that the generated example works:

```console
davis component validate ./my-component
davis model run ./my-component/examples/minimal/analysis.yaml
```

Then modify the calculation in `component.py`, the inputs, config, and outputs in `component.yaml`, and the example settings in `examples/minimal/analysis.yaml`. Install the finished package with:

```console
davis install component ./my-component
davis component inspect example/my-component
```

Validation checks the YAML structure, ID, version, schema, and referenced files. It cannot determine whether the calculation is scientifically correct, so retain a small test with known inputs and outputs. The component does not need to live in the Davis development repository.

For another language or a custom layout, generate a Manifest by specifying every command argument separately:

```console
davis component scaffold ./my-component \
  --id example/my-component \
  --kind transform \
  --command Rscript \
  --command component.R
```

The same form works for a native executable or another runtime. Davis does not install a general-purpose language runtime on behalf of the component.

The [Component Authoring Acceptance procedure](davis-component-authoring-acceptance.md) describes how to test the same contract through beginner, expert-YAML, and external-AI workflows.

### Asking an AI assistant to create a component

Give the AI this entire guide and at least:

1. The purpose and `kind` of the component.
2. Each input slot’s name, media type, and meaning.
3. Config fields and their constraints.
4. Each output artifact’s name, media type, and meaning.
5. Available runtime commands and required external environments.

Do not send confidential datasets to an AI service. Prefer a schema containing column names, types, units, missing-value rules, and category definitions but no records. If names are sensitive, use anonymized column names and a separate meaning map. A very small synthetic sample may be used only when it contains no real or identifying values.

Ask the AI not to invent Davis-specific fields and to provide both `davis component validate` and a known-input execution test. Review all generated code before executing it.

```text
Use only the attached Davis Component Authoring Guide as the source of
Davis-specific rules. Create a component for the following operation.

Purpose: (estimation or calculation)
Inputs: (file names, formats, and schema; do not include confidential records)
Settings changed per run: (variables, thresholds, and so on)
Outputs: (tables, metrics, figures, or reports)
Available language and commands: (Python, R, and so on)

Produce component.yaml, the implementation program, a minimal synthetic sample,
an Analysis Plan, and tests. Do not invent Davis-specific fields that are absent
from this guide. Include validation and sample-run commands.
```

## Manifest

The Manifest is the component’s instruction manual. Do not put values selected for an individual run directly in it. Define their shape under `configuration.schema`; selected values belong in the Analysis Plan.

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
    version: davis.ui/v1
    inputs:
      persons: {title: Persons, widget: table-binding}
    sections:
      - {bind: /person_id, widget: auto, title: Person ID column}
    results:
      - artifact: explanatory_variables
        title: Explanatory variables
        widget: table

outputs:
  artifacts:
    explanatory_variables:
      media_types: [text/csv]
      required: true
```

`kind` is `model`, `transform`, or `visualize`; omission means `model` for backward compatibility. The component defines its operation names. The author declares `requires_davis` from the Davis contract actually used, rather than copying the current Davis release number automatically.

`runtime.executor: process` is language-independent. If `command` reads `request.json` and writes `run-result.json`, Davis can run it in any language. Use `requirements` to declare required commands, optional SemVer constraints, and installation guidance. Davis verifies them before a run but does not install them. `version_arguments` defaults to `--version`.

Legacy `runtime.kind: python` and `runtime.kind: native` declarations remain readable as process runtimes.

### Declarative Desktop UI

A component with a Desktop form declares `presentation.ui.version: davis.ui/v1`. `inputs` describes input slots and `sections` maps parts of config to reusable widgets. `configuration.schema` remains authoritative; presentation only changes how values are edited or shown.

| Widget | Purpose |
| --- | --- |
| `table-binding` | Select one or more CSVs and configure a base table, join keys, relationship, and join type |
| `column-map` | Map semantic roles from the schema to input columns |
| `utility-terms` | Edit parameters, variable columns, constants, alternatives, and coefficients |
| `nests` | Edit alternative membership and fixed or estimated nest-scale values |
| `parameter-settings` | Edit initial values and bounds for parameters referenced by terms |
| `auto` | Generate controls for strings, numbers, booleans, and enums from JSON Schema |
| `extension:<id>` | Use a UI extension packaged with the component |
| unknown widget | Fall back to YAML for that section without disabling the whole form |

```yaml
presentation:
  ui:
    version: davis.ui/v1
    inputs:
      choice_data:
        title: Choice data
        widget: table-binding
        preparation: {component: davis/csv-transform, version: 0.4.1}
    sections:
      - {bind: /roles, widget: column-map, input: choice_data}
      - {bind: /terms, widget: utility-terms, input: choice_data}
      - bind: /nests
        widget: nests
        alternatives_from: /roles/alternative_id
      - {bind: /estimation, widget: auto}
```

`bind` is a JSON Pointer into `configuration.schema`; omitted `widget` means `auto`. A free-form object or array that cannot be rendered still has a section-level YAML editor. Multiple-CSV preparation is a shared Davis `table_binding`, not a model-specific feature.

For a bilingual Desktop presentation, write display strings under `component_name`, `title`, `description`, and `labels` as `{ja: ..., en: ...}` objects. A plain string remains valid and is shown in every language. If the selected language is absent, Desktop falls back to the other language. Keep the standard JSON Schema `title` and `description` fields as strings; use `x-davis-title` and `x-davis-description` for their bilingual Davis presentation values.

```yaml
configuration:
  schema:
    type: object
    properties:
      tolerance:
        type: number
        title: Tolerance
        x-davis-title: {ja: 収束判定値, en: Convergence tolerance}
presentation:
  ui:
    version: davis.ui/v1
    component_name: {ja: 到達可能性計算, en: Accessibility calculator}
    inputs:
      persons:
        title: {ja: 個人表, en: Person table}
    sections:
      - bind: /tolerance
        title: {ja: 推定設定, en: Estimation settings}
        widget: auto
```

### Component-packaged UI extensions

For an interaction that built-in widgets cannot express, a component can package a self-contained HTML fragment of at most 512 KiB and reference it as `extension:<id>`.

```yaml
presentation:
  ui:
    version: davis.ui/v1
    extensions:
      - id: nest-editor
        api_version: davis.widget/v1
        source: ui/nest-editor.html
    sections:
      - bind: /nests
        widget: extension:nest-editor
        context:
          alternatives:
            provider: distinct-values
            input: choice_data
            column_from: /roles/alternative_id
```

The fragment runs in a sandboxed `iframe` without direct network, parent-DOM, filesystem, or shell access. It may receive `render` and send only versioned `ready`, `set-value`, and `resize` messages. Values from `set-value` are still checked against the component JSON Schema.

Named context providers are:

| Provider | Declaration | Value supplied |
| --- | --- | --- |
| `config` | `path` | Another Analysis Plan config section |
| `columns` | `input` | Available columns and aliases for an input slot |
| `distinct-values` | `input`, `column_from` | Bounded distinct values, sample count, and truncation state |

Davis does not expose the input path or all rows to an extension. Resolution failures are delivered as named `context_errors` rather than stopping the entire editor.

Large definitions may use safe package-relative JSON or YAML references. Do not provide both an inline value and a reference.

```yaml
configuration:
  schema_ref: schemas/config.schema.json
presentation:
  ui_ref: schemas/ui.schema.json
```

For best portability between humans, web-based AI, and GUI editing, prefer a single inline `component.yaml` when its size remains manageable.

### Outputs and artifact profiles

`outputs.artifacts` restricts allowed artifacts and validates required artifacts and media types. Davis calculates the path, size, and BLAKE3 digest and records them in `result.json`.

`presentation.ui.results` can display a declared JSON object with `key-value` or a CSV with `table`. Unsupported or oversized artifacts remain available in the artifact list.

An optional `profile` gives an artifact a stable semantic role for common previews, Run comparison, and later components:

| Profile | Meaning | Supported forms |
| --- | --- | --- |
| `table` | General table | CSV, Parquet |
| `metrics` | Named metrics | JSON object, CSV, Parquet |
| `parameters` | Estimated parameters | CSV, Parquet |
| `predictions` | Predictions or choice probabilities | CSV, Parquet |
| `figure` | Figure or declarative figure specification | JSON, Vega-Lite JSON, HTML, PNG, SVG |
| `diagnostics` | Sample information, warnings, convergence | JSON object, CSV, Parquet |
| `report` | Human-readable report | HTML, Markdown, PDF, JSON |

A `parameters` CSV must contain `name` and `estimate`; `std_error`, `t_value`, `p_value`, `significance`, `lower`, and `upper` are optional. `t_value` denotes the Wald statistic `estimate / std_error`; official choice-model components calculate its p-value from the asymptotic normal distribution. Root values for JSON `metrics`, `diagnostics`, and `figure` artifacts must be objects. A profile is optional and does not prohibit custom artifacts.

## Analysis Plan

An Analysis Plan records one experiment. Create a different Plan when input files, variables, or settings change; do not edit `component.yaml` as an experiment log.

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

The legacy `model.component` form remains compatible. New Plans should use top-level `component` with its `id`.

## Process contract

Davis and the implementation program exchange two JSON files, so the interface is independent of programming language. Davis starts `runtime.command` in the component directory and appends the absolute request path after `request_argument`.

The component must:

1. Read `request.json`.
2. Read inputs only from `inputs.<name>.resolved.path`.
3. Write artifacts only below `output_directory`.
4. Write `output_directory/run-result.json`.
5. Exit with code 0 on success and a nonzero code on failure.

The request contains `api_version`, `run_id`, `operation`, resolved component identity, original and resolved input descriptions, validated `config`, and an absolute `output_directory`. Never assume input or output paths are relative to the component directory.

`run_id` is generated by Davis from `run.label`, or the Plan name, plus execution time and a unique suffix. Treat it as opaque and copy it unchanged to the result.

```json
{
  "api_version": "davis.result/v1alpha1",
  "run_id": "the same run ID as the request",
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

Artifact paths must be safe paths relative to `output_directory`. The component does not calculate size or digest; Davis derives them from the actual files.

## Connecting Runs

Pass a preprocessing output to another component by logical artifact identity instead of an absolute path:

```yaml
inputs:
  choice_data:
    kind: run_artifact
    run_id: run_123456
    artifact: transformed_table
```

Davis revalidates the prior Run’s status, artifact name, media type, size, digest, and safe path. At present, execute the first Run and insert its ID into the second Plan manually; there is no general pipeline DAG syntax.

## Binding multiple sources for estimation

Use `table_binding` on a model input when Davis should join and select columns immediately before estimation. Davis materializes the prepared table as Parquet under the same Run.

```yaml
inputs:
  choice_data:
    kind: table_binding
    processor:
      id: davis/csv-transform
      version: 0.4.1
    sources:
      choices: {kind: local, path: choices.csv}
      persons: {kind: local, path: persons.csv}
    base: choices
    joins:
      - source: persons
        relationship: many_to_one
        left_on: case_id
        right_on: person_id
    columns:
      travel_time: {source: choices, column: time}
      income: {source: persons, column: income}
```

Keys under `columns` are final names referenced by model config. Single and composite keys, `many_to_one` and `one_to_one`, left and inner joins, and explicit unmatched-key handling are supported. Nested bindings and joins from one auxiliary source to another are not supported.

For standalone joins and transformations, use the official `davis/csv-transform` component. It supports declared joins, linear column calculations, column selection, and CSV or Parquet output. The [Official Components User Guide](official-components-guide_en.md) contains executable MNL, NL, RL, and CSV Transform examples.

## Validation and distribution

```console
davis install component ./my-component
davis model validate analysis.yaml
davis model run analysis.yaml
davis component pack ./my-component --name my-component --out dist
davis component registry dist/my-component-0.1.0.entry.json \
  --out dist/component-registry.json
```

Do not include `.venv`, `__pycache__`, Git metadata, build output, or test caches in a published bundle. Lock dependencies and test successful input, missing columns, invalid values, and missing artifacts. A component can execute arbitrary code: install only trusted components. General sandboxing and registry signatures are not yet implemented.

## Acceptance with an AI unfamiliar with Davis

For the final portability test, give a new browser AI only this guide, the model requirements, a schema with no confidential records, and a small synthetic input sample. Do not provide Davis development conversations, source code, or an existing component to imitate.

The test passes when:

1. The AI creates `component.yaml` and an implementation program.
2. `davis component validate` succeeds.
3. The generated Analysis Plan can be understood and edited in the GUI.
4. The synthetic sample runs and Davis displays declared artifacts.
5. Missing columns and similar failures produce actionable messages.
6. Another person can explain the inputs, settings, and outputs from the Manifest, Plan, and result.

When the test fails, improve this guide, the scaffold, validator, or GUI at the point that caused the misunderstanding instead of relying on one AI’s prior knowledge.

## Current boundaries

Implemented features include local, Davis Catalog, and `run_artifact` inputs; multi-source table binding before estimation; declarative CSV joins, column selection, and linear combinations; CSV and Parquet output; language-independent process execution; artifact validation; and local or registry installation. Catalog inputs record `dataset_id` and `file_id` and resolve through the user-wide cache.

Revision pinning, declarative filtering and grouping, arbitrary pipeline DAGs, general sandboxing, and registry signatures are not implemented. Davis does not install Python or other general language environments. Data prepared manually in QGIS can be used as a local input; an automated algorithm can be packaged as a transform component under the same process contract.
