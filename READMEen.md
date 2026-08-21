# Davis

[日本語](README.md)

Davis is a platform that connects transport-data retrieval and behavioral-model execution in one workflow. The current implementation preserves the existing data-catalog capabilities while introducing content-addressed storage, a Rust CLI, and a Web catalog.

## Installation

Normal use does not require Rust, Cargo, or Python. The installer selects a prebuilt executable for the user's operating system and CPU architecture from GitHub Releases.

- [Installation Guide for Participants](docs/participant-installation_en.md) ([日本語](docs/participant-installation.md))
- [Installation Guide for Organizers](docs/operator-installation_en.md) ([日本語](docs/operator-installation.md))

Rust and Cargo are required only when developing or building Davis itself.

After a normal command completes, the CLI checks for a newer release at most once every 24 hours. When an update is available, it prints guidance without interrupting the command or corrupting JSON output. Run `davis update` to check explicitly and display the update command for the current operating system.

## Current structure

```text
crates/
  davis-core/       # DatasetManifest, content addressing, local cache
  davis-catalog/    # DVC and schema.yaml readers, catalog index generation
  davis-storage/    # Filesystem and S3-compatible storage
  davis-cli/        # list, info, index, verify, ingest, push, get

web/
  davis-web/        # Searchable Web catalog with schema filters and multi-selection

packages/
  dataset_cli/      # Legacy Python CLI

src/
  specific_model/  # Behavioral-model estimation and simulation code
  base_model/      # Transport simulation code
```

## Developing the Rust CLI

From an environment with Rust and Cargo:

```bash
cargo run -p davis-cli -- list
cargo run -p davis-cli -- info network/matsuyama
cargo run -p davis-cli -- get network/matsuyama
cargo run -p davis-cli -- get network/matsuyama --file link.csv
```

The official executable name is `davis`.

```bash
cargo install --path crates/davis-cli --locked --root ~/.local
davis --help
```

Participants sign in once with the participant code and the Davis Web URL supplied by the organizers. The CLI stores a revocable session and uses the authenticated Download API for `list`, `info`, and `get`.

```text
davis login https://davis-web.davis-bin.workers.dev
davis list
davis info network/matsuyama
davis get network/matsuyama
davis logout
```

Organizers use a separate organizer code. The restricted organizer session is valid for 30 days by default and removes the need to distribute R2 credentials.

```text
davis operator login https://davis-web.davis-bin.workers.dev
davis operator status
davis push network/matsuyama --dry-run
davis push network/matsuyama
davis operator logout
```

Large objects are sent to R2 through 32 MiB multipart uploads. Davis stores the revocable session, not the shared organizer code.

To verify current DVC metadata, create BLAKE3 objects, and update DatasetManifests during development:

```bash
cargo run -p davis-cli -- verify
cargo run -p davis-cli -- ingest --all
```

Use `davis push` to publish differential updates to R2 or a filesystem remote. A normal push ingests changed source files into the content-addressed store automatically, reuses unchanged files from the local cache, uploads missing objects, writes a revisioned CatalogIndex, and then switches `catalog/current.json`. Operators therefore do not need to run `ingest` separately during routine updates. `davis push --all --dry-run` checks all datasets without uploading or publishing, while `--rehash` re-reads every source file and verifies it against the DVC metadata.

## Web catalog

The Web catalog reads the current CatalogIndex from R2 and falls back to the static index included in the deployment before the first R2 publication. Developers can regenerate the static index from all `schema.yaml` files:

```bash
cargo run -p davis-cli -- index
cd web/davis-web
pnpm install
pnpm dev
```

The catalog supports text and column search, faceted filtering, Dataset and File details, raw YAML inspection, multi-selection, total-size calculation, and copying the corresponding `davis get` command. The Worker provides shared-code authentication, revocable sessions, short-lived download grants, private R2 delivery, organizer authentication, multipart uploads, and catalog publication APIs.

## Validation

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
uv run pytest -q

cd web/davis-web
pnpm test
pnpm lint
```

## License

The Davis software is released under the [MIT License](LICENSE). This software license does not apply to the datasets. Refer to the `license` field in each data file's corresponding `schema.yaml` for its terms of use.

## Documentation

- [Installation Guide for Participants](docs/participant-installation_en.md) ([日本語](docs/participant-installation.md))
- [Installation Guide for Organizers](docs/operator-installation_en.md) ([日本語](docs/operator-installation.md))
- [Davis specification](docs/davis-spec.md)
- [Platform concept](docs/davis-platform-concept.md)
- [Legacy dataset CLI](packages/dataset_cli/README_en.md)
- [Base model](src/base_model/README.md)

## Documentation update policy

When changing user-facing READMEs, installation guides, or operating guides, update the Japanese and English versions in the same commit or Pull Request. Do not update only one language first. Keep the relationship between each Japanese document and its English counterpart visible from the document itself or from its parent README.

For questions or contributions, contact the organizers of the Summer School on Behavioral Models.
