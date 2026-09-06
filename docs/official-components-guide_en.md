# Davis Official Components User Guide

[日本語](official-components-guide.md)

This guide is for users of the four official components distributed with Davis v0.5.3. It explains how to transform data or estimate models; it is not a guide for creating a new component. A user or an AI assistant can rely on this document as the usage specification without knowing the Davis development repository or its internal implementation.

## 1. Concepts

Davis separates a component from the settings of an individual analysis.

- `component.yaml`: the specification distributed by Davis. Ordinary users do not edit it.
- `model.yaml`, or another Analysis Plan filename: the files, columns, variables, and estimation settings for one analysis. A user or AI assistant creates and edits this file.
- `davis-runs/`: the result directory created by Davis.

When asking an AI assistant for help, provide this guide, the intended model or transformation, and a schema file that describes the columns you intend to use. The schema file should contain no actual records; include only information such as column names, data types, units, missing-value rules, and the meaning of categorical values. If a schema file is unavailable, provide only a header detached from the data. If column names may themselves disclose confidential information, use anonymized names together with a separate name-to-meaning mapping. Do not provide the CSV itself, actual values, personal names, identifiers, or other confidential information to an AI assistant. The AI assistant should create an Analysis Plan without modifying `component.yaml`.

## 2. Requirements

Update the Davis CLI to v0.5.3.

```console
davis update
davis --version
```

The official components use `uv 0.8` or later to start Python and the locked dependencies. Davis does not install `uv` itself.

```console
uv --version
```

If `uv` is unavailable, follow the [official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

## 3. Installing the official components

```console
davis install component mnl
davis install component nl
davis install component rl
davis install component csv-transform
davis installed
```

| Name | ID | Version | Purpose |
| --- | --- | --- | --- |
| Multinomial Logit | `davis/mnl` | `0.3.1` | Estimate an MNL from long-format choice data |
| Nested Logit | `davis/nl` | `0.1.1` | Estimate a two-level, non-overlapping NL |
| Recursive Logit | `davis/rl` | `0.1.1` | Estimate an RL from a link network and observed paths |
| CSV Transform | `davis/csv-transform` | `0.4.0` | Join CSVs, create linear-combination columns, select columns, and write CSV or Parquet |

When multiple CSV files are joined into a model input, Desktop uses `davis/csv-transform` internally. Install both the selected model and CSV Transform.

## 4. Common Desktop workflow

```console
davis install desktop
davis desktop
```

1. Select an empty or existing work folder under `Project workspace`. It does not need to be a Git repository.
2. Select the component under `ComponentManifest`.
3. Add a `Local CSV` or `Davis Catalog` file to each input.
4. For multiple CSVs, select the base table, left and right join keys, relationship, left or inner join, and whether unmatched rows are allowed.
5. Map role columns and configure variables and component-specific settings.
6. Inspect the Analysis Plan with `Preview YAML`, then save and run it.
7. Results remain both in the application and under `<workspace>/davis-runs/<run-id>/`.

A local CSV may remain outside the workspace. A Catalog input is recorded by `dataset_id` and `file_id` and resolved from the shared cache.

## 5. Running an Analysis Plan from the CLI

A Plan saved by the GUI and one written by a person or AI assistant use the same commands.

```console
davis model validate ./model.yaml
davis model plan ./model.yaml
davis model run ./model.yaml
```

Results are written to `davis-runs/` under the current directory by default. Use `--run-root` to choose another location. A local path in a Plan is resolved relative to the Plan file.

## 6. MNL

### Input

Assign one CSV or Parquet file to `choice_data`. Each row must represent one alternative in one choice situation.

| Role | Meaning |
| --- | --- |
| `case_id` | One column, or multiple columns, identifying a choice situation |
| `alternative_id` | Alternative ID of the row |
| `chosen` | A column marking the selected row, such as 0/1 |
| `chosen_alternative` | The selected alternative ID for the case; may replace `chosen` |
| `available` | Availability column; optional |
| `weight` | Case weight; optional |

Specify either `chosen` or `chosen_alternative`.

### Utility terms

Each term has a parameter name and either an input column or a constant. Omit `alternatives` to apply a term to every alternative, or list the alternatives to which it applies.

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

Estimation settings include `optimizer` (`bfgs` or `l-bfgs-b`), `max_iterations`, and `tolerance`. `development_case_limit` is only for a quick test and should be omitted from a full estimation.

### Minimal Plan

```yaml
api_version: davis.analysis/v1alpha1
name: mode-choice-mnl
component: {id: davis/mnl, version: 0.3.1, operation: estimate}
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

## 7. Nested Logit

The long-format input, `roles`, and `terms` are similar to MNL, but NL requires a `chosen` column and a `nests` configuration. Every alternative must belong to exactly one nest.

This component normalizes the top-level scale to 1 and uses `dissimilarity` as λ for each nest, with `0 < λ <= 1`. Use `initial` for an estimated λ or `fixed` for a fixed value; do not specify both. A singleton nest is fixed to λ=1 at runtime.

```yaml
api_version: davis.analysis/v1alpha1
name: mode-choice-nl
component: {id: davis/nl, version: 0.1.1, operation: estimate}
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

## 8. Recursive Logit

RL uses two tables.

- `network`: one directed link per row
- `observations`: a long-format table listing the traversed links of each trip in order

Required network roles are `link_id`, `from_node`, and `to_node`. Required observation roles are `trip_id`, `step`, `link_id`, and `destination`. `step` must order the observations uniquely within each trip, and `destination` is the destination node ID. Link and node IDs must be consistent between the two tables.

A utility term refers to a numeric column of the network table. `coefficient` is normally 1 and should only change when a sign or multiplier is fixed in advance. Each parameter may define `initial`, `lower`, and `upper`.

```yaml
api_version: davis.analysis/v1alpha1
name: route-choice-rl
component: {id: davis/rl, version: 0.1.1, operation: estimate}
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

## 9. CSV Transform

CSV Transform is not an estimator. It reproducibly processes input data with the following operations:

- `joins`: left or inner joins using a single or composite key
- `calculations`: linear combinations of columns and constants
- `select`: final column selection and renaming
- `output`: CSV or Parquet format and compression

The join `relationship` is `many_to_one` or `one_to_one`; unexpected duplicates are errors. `allow_unmatched` defaults to `false`.

```yaml
api_version: davis.analysis/v1alpha1
name: prepare-choice-data
component: {id: davis/csv-transform, version: 0.4.0, operation: transform}
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

You may run CSV Transform first as a separate Run. Alternatively, add multiple CSVs to a model input in Desktop and let Davis execute it as the shared `table_binding` immediately before estimation.

## 10. Results and troubleshooting

When available, model components return:

- `parameters.csv`: parameter names, estimates, standard errors, and related statistics
- `metrics.json`: log-likelihood, AIC, BIC, convergence information, and related metrics
- `predictions.csv`: predicted probabilities or observed-link choice probabilities
- `sample-summary.json`: used and excluded cases and warnings

CSV Transform returns `transformed.csv` or `transformed.parquet` and a transformation summary. The artifact list in each Run's `result.json` is authoritative.

When a run fails, check:

1. `davis --version` is 0.5.3 or later.
2. `uv --version` is 0.8 or later.
3. `davis installed` lists every required component.
4. Every Plan column name exactly matches the CSV header.
5. Choices within each case, join keys, and IDs shared by the network and observations are consistent.

To create a new component instead of using these four, refer to the [Davis Component Authoring Guide](davis-component-authoring.md).
