# Davis

[日本語](README.md)

Davis is a platform that connects transport-data retrieval and behavioral-model execution in one workflow. The current implementation preserves the existing data-catalog capabilities while introducing content-addressed storage, a Rust CLI, and a Web catalog.

## Installation

Normal use does not require Rust, Cargo, or Python. The installer selects a prebuilt executable for the user's operating system and CPU architecture from GitHub Releases.

- [Installation Guide for Participants](docs/participant-installation_en.md) ([日本語](docs/participant-installation.md))
- [Installation Guide for Organizers](docs/operator-installation_en.md) ([日本語](docs/operator-installation.md))

For organizers, the guide's first five sections cover installation, repository, session and personal-branch setup, dataset updates, review, and publication. Both terminal and VS Code procedures are included.

Rust and Cargo are required only when developing or building Davis itself.

After a normal command completes, the CLI checks for a newer release at most once every 24 hours. The source of truth for update metadata is `latest-version.json`, attached to each GitHub Release, so a CLI release does not require a Web redeployment. When an update is available, Davis prints guidance without interrupting the command or corrupting JSON output. Run `davis update` to review the release and answer `y/N`; after approval, Davis runs the installer for the current operating system. Use `davis update --yes` to skip confirmation.

## Current structure

```text
crates/
  davis-core/       # DatasetManifest, content addressing, local cache
  davis-catalog/    # Davis Manifest and schema.yaml readers, catalog index generation
  davis-document/   # Deterministic Japanese and English PDFs from schemas
  davis-storage/    # Filesystem and S3-compatible storage
  davis-cli/        # list, info, index, verify, ingest, push, get

web/
  davis-web/        # Searchable Web catalog with schema filters and multi-selection

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

The legacy Python CLI is no longer shipped in current releases. For investigation or recovery, see the [`legacy-python-final` tag](https://github.com/bin-utokyo/davis/tree/legacy-python-final/packages/dataset_cli) or the `legacy/python-cli-v0` branch.

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
davis pull network/matsuyama
davis logout
```

Use `get` for a first retrieval or selective file retrieval. Use `pull` either for the first full retrieval of a dataset or to synchronize it to the current Manifest. Because `pull` updates existing files with remote contents, do not run it while local edits remain.

Plain `davis pull` and `davis push` without a dataset ID select every dataset. Supplying a dataset ID selects only that dataset. `davis push --all` remains available as a compatibility alias for the same full operation.

Organizers use a separate organizer code. The restricted organizer session is valid for 30 days by default and removes the need to distribute R2 credentials.

```text
davis operator login https://davis-web.davis-bin.workers.dev
davis operator status
davis push network/matsuyama --dry-run
davis push network/matsuyama
davis publish
davis operator logout
```

Large objects are sent to R2 through 32 MiB multipart uploads. Davis stores the revocable session, not the shared organizer code.

To verify local real data against the BLAKE3 IDs in the current Davis Manifests:

```bash
cargo run -p davis-cli -- verify
cargo run -p davis-cli -- ingest --all
```

With an official operator session, `davis push <dataset>` from any personal working branch other than `main` hashes new, changed, or uncached files and reuses unchanged files from the previous Manifest and local cache. Davis does not prescribe a branch-name format. After missing R2 objects upload successfully, it generates only the Japanese and English PDFs affected by a changed schema or object ID, stages and commits the selected dataset's schemas, PDFs, and Manifest, then pushes the current personal branch to GitHub. It does not publish the Catalog. Use `--dry-run` first; it does not change the repository, cache, PDFs, R2, or Git. Use `--rehash` only to re-read every selected file. The normal path does not use DVC or `.dvc` files.

Without an operator session, a direct filesystem or S3-compatible remote configured in `.davis/config.toml` synchronizes objects and local Manifests without requiring the official branch name, `origin/main`, or GitHub. It does not create or push a Git commit. This preserves the same Manifest and object format for independent repositories, MinIO, S3, and local storage.

After the metadata Pull Request has been reviewed and merged, run `davis publish` from a clean, current `main` to update the Web catalog. The command verifies the branch, its exact match with `origin/main`, the working tree, and R2 object coverage before writing a revisioned CatalogIndex and switching `catalog/current.json`. Personal branches can synchronize objects in advance but cannot publish the Catalog.

By default, `davis get` and `davis pull` save each data file together with its corresponding `schema.yaml`. Japanese and English PDF documentation can be added independently, while omitting schemas requires an explicit option.

```bash
davis get routes/Matsuyama
davis get routes/Matsuyama --pdf-ja
davis get routes/Matsuyama --pdf-en
davis get routes/Matsuyama --pdf-ja --pdf-en
davis get routes/Matsuyama --no-schema
davis pull routes/Matsuyama
davis pull routes/Matsuyama --pdf-ja --pdf-en
davis pull
```

When selected files already exist at the `get` destination, Davis asks whether to replace them. Enter `y` to replace them all, or enter `N` (or press Enter) to cancel without changing them. Use `--force` to skip the confirmation in automated use. In a non-interactive environment, Davis stops unless `--force` is specified to prevent unintended replacement.

Git remains the source of truth for `schema.yaml` and the PDF documentation; these files are not duplicated as R2 objects. The CatalogIndex records the schema contents and GitHub URLs for available PDFs. CLI and Web clients save schemas from the Catalog API and retrieve PDFs from GitHub, while only the actual data is delivered as private R2 objects.

## Web catalog

The Web catalog reads the current CatalogIndex from R2 and falls back to the static index included in the deployment before the first R2 publication. Developers can regenerate the static index from all `schema.yaml` files:

```bash
cargo run -p davis-cli -- index
cd web/davis-web
pnpm install
pnpm dev
```

The catalog supports text and column search, faceted filtering, Dataset and File details, raw YAML inspection, multi-selection, total-size calculation, and copying the corresponding `davis get` command. The download dialog selects `schema.yaml` by default and lets users add Japanese and English PDFs independently. An inline warning explains the effect of omitting schemas from future formatting and modeling workflows. The Worker provides shared-code authentication, revocable sessions, short-lived download grants, private R2 delivery, organizer authentication, multipart uploads, and catalog publication APIs.

The official Worker records each request reaching the object-delivery endpoint with a valid Download Grant as a download attempt in Cloudflare Analytics Engine. It stores only the File ID, path, Object ID, and full/range distinction; it does not store personal information or session tokens. Analytics failures never fail the download. This adds no changes to the public API, CLI, Manifest, or schema formats.

## Validation

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

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
- [Base model](src/base_model/README.md)

## Documentation update policy

When changing user-facing READMEs, installation guides, or operating guides, update the Japanese and English versions in the same commit or Pull Request. Do not update only one language first. Keep the relationship between each Japanese document and its English counterpart visible from the document itself or from its parent README.

For questions or contributions, contact the organizers of the Summer School on Behavioral Models.
