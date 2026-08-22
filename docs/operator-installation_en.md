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

## Preparing an organizer session

Specify the Davis Web URL provided by the organizer team and enter the shared organizer code at the interactive prompt. The code does not need to appear in a command argument or shell history.

```text
davis operator login <Davis Web URL>
davis operator status
```

Davis does not store the shared code itself. It stores only a revocable organizer session, so authentication normally does not need to be repeated while that session remains valid. `push` and `publish` use the organizer session, while `get` and `pull` require a separate download-only participant session. An organizer who retrieves real data should also run `davis login <Davis Web URL>` once.

## Git operations and Davis operations

Git records the history of code, schemas, PDFs, and DatasetManifests. R2 stores immutable real-data objects. Real data is not committed to Git. Davis synchronizes both systems in a safe order.

### Git commands used in this guide

| Command | Purpose | Effect on real data or R2 |
| --- | --- | --- |
| `git clone <URL>` | Copy the repository to a PC for the first time | None |
| `git status` | Inspect the branch and uncommitted changes | None |
| `git switch <branch>` | Switch between `main` and the personal working branch | None |
| `git pull --ff-only` | Safely fast-forward the current branch to GitHub | None |
| `git fetch origin` | Retrieve current GitHub history without switching branches | None |
| `git merge --ff-only origin/main` | Safely fast-forward the personal branch to current `main` | None |

Routine data updates do not require separate `git add`, `git commit`, or `git push` commands. With an official organizer session, `davis push` performs them for only the selected dataset. Git commands alone do not retrieve, upload, or publish real data.

### Davis commands used by organizers

| Command | Purpose | Effect on the public Web catalog |
| --- | --- | --- |
| `davis --version` | Show the installed version | None |
| `davis update` | Check the latest release and update method | None |
| `davis login <URL>` / `logout` | Store or remove the participant download session | None |
| `davis operator login <URL>` / `status` / `logout` | Manage the organizer upload and publication session | None |
| `davis list` | List available datasets | None |
| `davis info <dataset>` | Inspect files, sizes, and schema coverage | None |
| `davis get <dataset>` | Retrieve a dataset or selected files for the first time | None |
| `davis pull <dataset>` | Synchronize a whole dataset to its published Manifest | None |
| `davis pull` | Retrieve or synchronize every dataset | None |
| `davis verify [dataset]` | Compare local real data with the BLAKE3 IDs in the current Davis Manifest | None |
| `davis push <dataset> --dry-run` | Inspect planned objects and bytes | None |
| `davis push <dataset> [-m <message>]` | Prepare and send one assigned dataset to R2 and the personal branch | None |
| `davis push` / `davis push --all` | Inspect and send every dataset to R2 and the personal branch | None |
| `davis publish` | Publish reviewed, current `main` | Yes |

`davis ingest`, `davis documents`, and `davis index` are development and maintenance commands. Routine updates do not use them. A normal `push` reuses unchanged files when their previous Manifest entry and local cache object remain valid, and hashes only new, changed, or uncached files. Use `--rehash` only to read every selected file again.

For `get`, repeat `--file` to select files or directories and use `--pdf-ja` and `--pdf-en` to include documentation PDFs. Schemas are saved by default and omitted only with `--no-schema`. `pull` provides the same document options. Run `davis <command> --help` for the complete option list.

### Sources of truth and generated artifacts

The personal contributor edits only the assigned dataset's real data and `schema.yaml`. Davis detects real files directly below the dataset root and does not use `.dvc`. Additions, modifications, renames, moves, and removals are reflected in the DatasetManifest by the next `push`.

The Japanese and English PDFs and the DatasetManifest are derived artifacts. After the R2 upload succeeds, a normal `davis push` deterministically generates only the PDFs affected by a changed schema or BLAKE3 object ID. The contributor does not edit PDFs or Manifests manually and does not run `git add`, `git commit`, or `git push` separately.

A normal `davis push` performs these steps in order:

1. Validate the personal branch and its relationship to `origin/main`.
2. Reject uncommitted changes outside the selected dataset.
3. Reuse unchanged files from the previous Manifest and local cache, then generate the required BLAKE3 objects and DatasetManifest.
4. Upload missing objects to R2.
5. After upload succeeds, generate only the Japanese and English PDFs affected by a changed schema or object ID.
6. Stage only the selected dataset's schemas, PDFs, and Manifest.
7. Commit and push the current personal branch to GitHub.

If a step fails, later steps do not run. R2 objects are immutable and content-addressed, so an object uploaded before a Git failure does not damage the current publication. The public Catalog remains unchanged until `davis publish`.

### Personal working branches

Organizers update data on personal working branches rather than `main`. Davis does not require a branch prefix or a match with a GitHub username; use any name your team can identify. Reusing one personal branch is recommended to avoid clutter, but creating a new branch is also supported.

Create the personal branch from current `main` once:

```bash
git status
git switch main
git pull --ff-only
git switch -c <personal-working-branch>
git push -u origin <personal-working-branch>
```

Before every later edit, fast-forward the personal branch to current `main`:

```bash
git status
git switch <personal-working-branch>
git fetch origin
git merge --ff-only origin/main
```

If uncommitted changes exist or `--ff-only` fails, do not reset, stash, rebase, or force a merge. Preserve the work and consult the organizer team. Merge Pull Requests with a merge commit and retain the personal branch. Do not use squash or rebase merge because either prevents the next `--ff-only` update.

`davis push` accepts any named branch other than `main`. It rejects a detached HEAD and direct execution from `main`. `davis publish` cannot run from a personal branch.

### Standard workflow for one dataset update

1. Switch to the persistent personal branch and fast-forward it to current `main`.
2. Only when the published real data is not already available locally, run `davis pull <dataset>` before editing. Do not run `pull` after editing because it overwrites local changes.
3. Edit the assigned dataset's real data and `schema.yaml`. Describe the intent in the Pull Request when adding, moving, renaming, or removing files.
4. Run a dry run. It reuses unchanged files and reads only the files that require hashing. It does not modify the repository, cache, PDFs, R2, or Git.

```bash
davis push routes/Matsuyama --dry-run
```

5. Review `Missing objects`, `Existing objects`, and `Upload size`. Do not continue when they differ from the intended update.
6. Run the normal push with a commit message. When omitted, the message defaults to `data: update <dataset>`.

```bash
davis push routes/Matsuyama -m "data: update routes/Matsuyama"
```

7. Confirm `Objects synchronized: yes`, `Git branch pushed: operator/...`, and `Catalog published: no`. If the command fails, do not repeatedly retry it. Share the complete displayed error with the organizer team.
8. Open a Pull Request from the personal branch to `main`. Another organizer reviews the schemas, PDFs, file layout, DatasetManifest, and expected size.
9. Merge the Pull Request with a merge commit and retain the personal branch.
10. Assign one publisher, switch to current `main`, and publish. The publisher does not need the real data locally.

```bash
git switch main
git pull --ff-only
git status
davis publish
```

11. Confirm `Catalog published: yes`, force-refresh the Web catalog, and verify the name, schema, license, file count, PDFs, and download.

### Why production publications are serialized

The CatalogIndex represents the current state of all Davis datasets, and `catalog/current.json` points to exactly one revision. If two organizers publish concurrently from different branches or stale copies of `main`, the operation that finishes last can replace the earlier catalog and make another dataset update disappear from the Web interface. Content-addressed R2 objects are normally not lost, but the published catalog may stop referring to them.

For every production publication:

- Assign exactly one publisher.
- Publish only from the latest `main`.
- Confirm that the working tree is clean.
- Confirm that the relevant Pull Request has been merged.
- Wait for any active publication to finish before starting another.
- Supply a dataset ID to synchronize one dataset. Use plain `davis push` or its compatibility alias `davis push --all` for all datasets.

When Pull Requests for several datasets are ready at nearly the same time, each organizer can synchronize objects independently from a personal branch. After all object synchronization and reviews are complete, merge the Pull Requests and have the designated publisher run `davis publish` once from the latest `main`.

### Why `push` and `publish` are separate

A multi-organizer workflow requires separate object synchronization and publication operations:

```text
davis push <dataset>   # Synchronize PDFs, Manifest, R2, and Git on a personal branch
davis publish          # Publish only the CatalogIndex from reviewed, current main
```

Content-addressed object uploads are deduplicated and remain invisible to participants until a catalog refers to them. They can therefore be performed safely from a personal branch. Publication changes participant-visible state and must be restricted to the current `main` and the designated publisher.

Davis separates these operations. `davis push` is safe to use on a personal branch, while `davis publish` is restricted to reviewed, current `main`. The publisher can use the organizer session and does not need R2 credentials.

### Recovering from an incorrect publication

If an old CLI or disabled guard allows publication from a personal branch or stale `main`, do not delete R2 objects. The current CLI rejects this operation. Notify the organizer team, then republish from one machine holding the correct, current `main`.

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

## Using storage outside the official deployment

The personal-branch, `origin/main`, and Pull Request rules in this guide are the operating policy for safely maintaining the official Davis Catalog as a team. When no organizer session is active, `davis push` to a filesystem or S3-compatible remote configured in `.davis/config.toml` does not require the official branch name or GitHub and does not create or push a Git commit. Another organization can retain the common Object and Manifest formats while defining its own review and publication policy.

## Troubleshooting

- If `davis operator status` reports an expired session, run `davis operator login <URL>` again.
- If `davis` is not found, open a new terminal.
- If the legacy CLI still runs, use `which -a davis` on macOS or Linux, or `Get-Command davis -All` on Windows, to inspect executable paths.
- If migration fails while Git has uncommitted changes, do not force-reset the repository. Consult the organizer team.
