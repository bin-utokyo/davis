# Davis Installation Guide for Organizers

[日本語](operator-installation.md)

This guide is for organizers who manage Davis metadata and publish data and catalogs to R2. If you only retrieve data, use the [Installation Guide for Participants](participant-installation_en.md).

## Requirements

- Git
- GitHub access to the Davis repository
- The shared organizer code distributed through an organizer-only channel

Routine organizer operations do not require Rust, Cargo, Python, DVC, Google Drive, or R2 credentials. Rust and Cargo are required only when developing or building Davis itself.

## Installing the Davis CLI

### macOS and Linux

```bash
curl --proto '=https' --tlsv1.2 -fsSL https://raw.githubusercontent.com/bin-utokyo/davis/main/scripts/install.sh | sh
```

### Windows

Run the following command in PowerShell:

```powershell
irm https://raw.githubusercontent.com/bin-utokyo/davis/main/scripts/install.ps1 | iex
```

The installer detects the operating system and CPU architecture and installs the matching macOS, Windows, or Linux executable. If the legacy Python CLI was installed through `uv`, the installer removes it before replacing `davis` with the current CLI.

Open a new terminal and verify the installation:

```text
davis --version
davis operator --help
```

## Preparing the repository

Clone the repository for a first-time setup:

```bash
git clone https://github.com/bin-utokyo/davis.git
cd davis
```

If you already have a clone and have not started a data update, confirm that the working tree is clean and update to the current `main` branch:

```bash
git status
git switch main
git pull --ff-only
```

If `git status` reports uncommitted changes, do not delete, stash, or force-update them. Review the work and consult the organizer team before proceeding.

## Git operations and Davis operations

Git manages repository code and metadata. Davis retrieves, updates, validates, synchronizes, and publishes transport data. Git commands cannot retrieve the real datasets. Use `davis get` for new or selective retrieval, and use `davis pull` either for the first full retrieval or to synchronize a whole dataset to its current Manifest.

### Git commands used in this guide

| Command | Purpose | Effect on real data |
| --- | --- | --- |
| `git clone <URL>` | Copy the repository to a PC for the first time | Does not retrieve real data |
| `git status` | Show the current branch and uncommitted changes | Makes no changes |
| `git switch <branch>` | Change the working branch | Git-untracked real data normally remains in place |
| `git pull --ff-only` | Retrieve current code and metadata commits from GitHub | Does not retrieve real data |
| `git fetch origin` | Retrieve current commit information without changing branches | Does not retrieve real data |
| `git merge --ff-only origin/main` | Safely fast-forward a personal branch to current `main` | Does not retrieve real data |
| `git add` and `git commit` | Record changes to `.dvc`, YAML, PDFs, Manifests, and related files | Do not commit the real data itself |
| `git push` | Send personal-branch commits to GitHub | Does not change R2 or the public Web catalog |

`git push` is not a Davis feature. It is documented because its name can otherwise be confused with `davis push`.

### Davis commands used by organizers

Run these commands from the repository root (the `davis` directory) unless stated otherwise.

| Command | Purpose | Primary state changed | Effect on public Web |
| --- | --- | --- | --- |
| `davis --version` | Show the installed version | None | None |
| `davis update` | Check the current release and update instructions | None | None |
| `davis login <Web URL>` | Save a participant download session | Local session | None |
| `davis logout` | Remove the participant session | Local session | None |
| `davis operator login <Web URL>` | Save an organizer upload and publish session | Local session | None |
| `davis operator status` | Check the organizer session | None | None |
| `davis operator logout` | Remove the organizer session | Local session | None |
| `davis list` | List available datasets | None | None |
| `davis info <dataset>` | Show files, sizes, and schema coverage | None | None |
| `davis get <dataset>` | Retrieve a whole dataset or selected files | Local `data/...` | None |
| `davis pull <dataset>` | Retrieve a whole dataset for the first time or synchronize existing files to the current Manifest | Local `data/...` | None |
| `davis pull` | Retrieve or synchronize every dataset | Local `data/...` | None |
| `davis verify [dataset]` | Validate local data against `.dvc` metadata | None | None |
| `davis push <dataset> --dry-run` | Show objects and bytes planned for R2 synchronization | None | None |
| `davis push <dataset>` | Synchronize missing dataset objects and update the DatasetManifest | R2 objects and local Manifest | None |
| `davis push` | Validate every dataset and synchronize all missing objects | R2 objects and local Manifests | None |
| `davis publish` | Publish the current `main` CatalogIndex | R2 Catalog revision | Yes |

`davis ingest` and `davis index` are development and maintenance commands. Routine updates do not run them separately because `davis push` performs the required ingestion and Manifest update internally.

### Common `get`, `pull`, and `push` options

| Example | Behavior |
| --- | --- |
| `davis get routes/Matsuyama` | Retrieve the whole dataset and its `schema.yaml` into the standard hierarchy |
| `davis get routes/Matsuyama --file <file-id-or-directory>` | Retrieve only the specified file or directory prefix; repeat `--file` for multiple selections |
| `davis get routes/Matsuyama --pdf-ja --pdf-en` | Retrieve available Japanese and English PDFs in addition to the default schema |
| `davis get routes/Matsuyama --no-schema` | Retrieve only the real data without saving schemas |
| `davis get routes/Matsuyama --out <directory>` | Recreate `data/routes/Matsuyama/...` below the specified directory |
| `davis pull routes/Matsuyama` | Retrieve the whole dataset, replacing existing files with the current Manifest contents |
| `davis pull routes/Matsuyama --pdf-ja --pdf-en` | Save or update schemas and available Japanese and English PDFs during synchronization |
| `davis pull` | Retrieve or synchronize every dataset |
| `davis push routes/Matsuyama --dry-run` | Show differences and planned bytes without uploading |
| `davis push routes/Matsuyama --rehash` | Re-read and validate the selected files instead of reusing the previous record |
| `davis push` or `davis push --all` | Validate and synchronize every dataset; do not use this for routine assigned updates |

Run `davis <command> --help` for the complete argument list, for example `davis get --help`, `davis pull --help`, or `davis push --help`.

## Participant login

Organizers who retrieve data through the CLI also sign in with the participant code:

```text
davis login https://davis-web.davis-bin.workers.dev
```

The participant session is download-only.

To retrieve the assigned real dataset for the first time, run either command below from the repository root. `git pull` cannot retrieve real data. `pull` is usually clearer when an organizer maintains and synchronizes the whole dataset; use `get` for selective retrieval.

```text
davis get routes/Matsuyama
# or
davis pull routes/Matsuyama
```

The default destination is `./data/routes/Matsuyama/...`. Objects already available locally are reused from the cache.

## Organizer login

```text
davis operator login https://davis-web.davis-bin.workers.dev
davis operator status
```

Enter the organizer-only shared code at `Operator code:`. Davis does not store the code itself. It stores a restricted organizer session that is valid for 30 days by default. You can run `push` and `publish` without entering the code again during that period. After expiration, the next interactive `push` or `publish` asks for the code once and renews the session.

## Checking connectivity

Run a dry run for the dataset you maintain from the repository root:

```text
davis push routes/Matsuyama --dry-run
```

The setup is ready when the output shows `Remote: ... (operator session)` together with the object counts and planned upload size. `--dry-run` does not modify R2 or the published catalog.

## Operating principles

### Storage responsibilities of Git and Davis

Organizer operations use two different commands named `push`. Keep their effects separate; refer to the earlier command reference for details.

| Operation | Destination | Main content | Effect on the public Web catalog |
| --- | --- | --- | --- |
| `git push` | GitHub | Code, `.dvc`, `schema.yaml`, PDF documentation, and DatasetManifests | Opening a Pull Request does not change the public catalog |
| `davis push` | R2 | Immutable data objects for the assigned dataset | Does not change the public Web catalog |
| `davis publish` | Davis Web API | A CatalogIndex generated from reviewed `main` | Switches the public Web catalog |

Treat Git as the source of truth for metadata and documentation, and R2 as the object store for real data. `schema.yaml` and both PDFs remain only in Git and are not duplicated in R2. The CatalogIndex contains the YAML content needed for search and GitHub references to the PDFs.

### One persistent working branch per organizer

Each organizer maintains one personal working branch and reuses it instead of creating a new branch for every update. Do not commit directly to `main`. The recommended name is `operator/<GitHub-username>`.

Create and register the branch from current `main` once during initial setup:

```bash
git status
git switch main
git pull --ff-only
git switch -c operator/<GitHub-username>
git push -u origin operator/<GitHub-username>
```

For every later update, fast-forward the same personal branch to current `main` before editing. Run `davis pull` only after switching to the personal branch. Because `davis pull` may update Git-managed schemas or PDFs, running it on `main` could leave uncommitted changes there.

```bash
git status
git switch operator/<GitHub-username>
git fetch origin
git merge --ff-only origin/main
```

If `git status` shows uncommitted work or `git merge --ff-only origin/main` fails, do not force a merge, rebase, reset, or stash the work. Preserve it and consult the organizer team.

To make this persistent-branch workflow safe, merge these Pull Requests with a **merge commit**. Do not use squash merge or rebase merge: they do not retain the personal branch commits as ancestors of `main`, so the next `--ff-only` update would fail. Do not delete the personal branch after merge.

On a personal branch, you may:

- Retrieve and edit the dataset assigned to you.
- Update `.dvc`, `schema.yaml`, the Japanese and English PDFs, and DatasetManifests.
- Run `davis verify`.
- Inspect planned changes with `davis push <dataset> --dry-run`.
- Commit the Git-managed files, run `git push`, and open a Pull Request.

You may run `davis push` from a personal branch. It synchronizes only content-addressed objects to R2 and does not change the public Catalog. Always specify the assigned dataset and inspect it with `--dry-run` first. `davis publish`, which changes participant-visible state, cannot run from a personal branch. The CLI rejects it unless the branch, working tree, and `origin/main` state are valid.

### Standard workflow for one dataset update

1. Follow the preceding steps to switch to the persistent personal branch and fast-forward it to current `main`.
2. On the personal branch, run `davis pull <dataset>` to retrieve the assigned dataset for the first time or synchronize it to the current version. Use `davis get <dataset>` when selectively retrieving new files. If local edits may remain, inspect them before allowing synchronization to overwrite anything. Then update the real data, `.dvc`, and `schema.yaml`.
3. Generate the Japanese and English PDFs from the YAML and include all three in the same Git commit.
4. Validate only the affected dataset.

```bash
davis verify routes/Matsuyama
davis push routes/Matsuyama --dry-run
git status
```

5. Confirm that no unintended dataset or file is included, review the planned upload size, and check that the schema and PDFs agree.
6. Synchronize the assigned dataset objects from the personal branch. This does not change the participant-facing Web catalog.

```bash
davis push routes/Matsuyama
```

7. Confirm `Objects synchronized: yes` and `Catalog published: no`. Commit only the changed Git-managed files, including the DatasetManifest updated by `davis push`, and send them to GitHub. Do not use `git add .` or `git add data/`, because either can accidentally stage real data or unrelated work. Name each file explicitly. The following is an example; omit files that did not change.

```bash
git status
git add .davis/datasets/routes/Matsuyama.yaml
git add data/routes/Matsuyama/path.csv.dvc
git add data/routes/Matsuyama/path.csv.schema.yaml
git add data/routes/Matsuyama/path.csv.ja.pdf
git add data/routes/Matsuyama/path.csv.en.pdf
git status
git commit -m "data: update routes/Matsuyama"
git push
```

8. Open a Pull Request from the personal branch to `main` on GitHub.
9. Another organizer reviews the column definitions, terms of use, year, filenames, moves or removals, DatasetManifest, and expected upload size.
10. Merge the Pull Request into `main` with a merge commit, not a squash or rebase merge. Keep the personal branch for the next update.
11. Assign one publisher and confirm that no other publication is running. The publisher does not need the real data locally because step 6 has already synchronized the required objects.
12. On the publisher's machine, switch to the latest `main`.

```bash
git switch main
git pull --ff-only
git status
```

13. Confirm that the working tree is clean, then publish. `davis publish` independently requires `main`, a clean working tree, and an exact match with `origin/main`. It also refuses publication if any Catalog object is missing from R2.

```bash
davis publish
```

14. Confirm `Catalog published: yes`, force-refresh the Web catalog, and verify the name, schema, license, file count, and download.

### Why production publications are serialized

The CatalogIndex represents the current state of all Davis datasets, and `catalog/current.json` points to exactly one revision. If two organizers publish concurrently from different branches or stale copies of `main`, the operation that finishes last can replace the earlier catalog and make another dataset update disappear from the Web interface. Content-addressed R2 objects are normally not lost, but the published catalog may stop referring to them.

For every production publication:

- Assign exactly one publisher.
- Publish only from the latest `main`.
- Confirm that the working tree is clean.
- Confirm that the relevant Pull Request has been merged.
- Wait for any active publication to finish before starting another.
- Normally specify the assigned dataset for object synchronization. Reserve `davis push` without an ID, or `davis push --all`, for explicit full validation or bulk synchronization.

When Pull Requests for several datasets are ready at nearly the same time, each organizer can synchronize objects independently from a personal branch. After all object synchronization and reviews are complete, merge the Pull Requests and have the designated publisher run `davis publish` once from the latest `main`.

### Why `push` and `publish` are separate

A multi-organizer workflow requires separate object synchronization and publication operations:

```text
davis push <dataset>   # Synchronize immutable objects from a personal branch
davis publish          # Publish only the CatalogIndex from reviewed, current main
```

Content-addressed object uploads are deduplicated and remain invisible to participants until a catalog refers to them. They can therefore be performed safely from a personal branch. Publication changes participant-visible state and must be restricted to the current `main` and the designated publisher.

Davis separates these operations. `davis push` is safe to use on a personal branch, while `davis publish` is restricted to reviewed, current `main`. The publisher can use the organizer session and does not need R2 credentials.

### Recovering from an incorrect publication

If someone publishes from a personal branch or stale `main`, do not delete R2 objects. Notify the organizer team, then republish from one machine holding the correct, current `main`.

```bash
git switch main
git pull --ff-only
git status
davis publish
```

If incorrect metadata has already been merged into `main`, review and merge a corrective or revert Pull Request before republishing. Deleting R2 objects is not a recovery operation. Clean up unreferenced objects later under the separate retention and deletion-approval procedure.

### Credentials and secrets

- Never place the organizer code in the repository, a commit, an issue, a Pull Request, a command-line argument, or a message with public recipients.
- Never add session information under `.davis` to Git.
- If the organizer code leaks, rotate both the shared code and organizer access revision to invalidate all existing organizer sessions.
- Do not distribute R2 credentials to routine organizer machines.

Adding, moving, renaming, or deleting a data file affects Catalog IDs and reproducibility. Do not treat these changes as routine content edits. Explain their purpose and impact in the Pull Request and obtain review. Never upload objects or publish the Catalog only to test installation.

## Updating

The CLI checks for a newer release once every 24 hours and displays a notice after a normal command when an update is available. Run `davis update` to compare the installed version with the latest release and display the update command for the current operating system. Run the displayed installer to update. The installer preserves the repository, real data, participant session, and organizer session.

## Troubleshooting

- If `davis operator status` reports an expired session, run `davis operator login <URL>` again.
- If `davis` is not found, open a new terminal.
- If the legacy CLI still runs, use `which -a davis` on macOS or Linux, or `Get-Command davis -All` on Windows, to inspect executable paths.
- If migration fails while Git has uncommitted changes, do not force-reset the repository. Consult the organizer team.
