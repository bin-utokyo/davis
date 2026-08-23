# Davis Installation Guide for Organizers

[日本語](operator-installation.md)

This guide is for organizers who manage Davis metadata and publish data and catalogs to R2. If you only retrieve data, use the [Installation Guide for Participants](participant-installation_en.md).

## Read these five sections first

Routine organizer work is fully covered by the following five sections. A first-time organizer should read them in order. Detailed command references, safety rationale, exceptional cases, and publication recovery follow afterward.

1. [Install the Davis CLI](#1-install-the-davis-cli)
2. [Prepare the repository](#2-prepare-the-repository)
3. [Prepare an organizer session](#3-prepare-an-organizer-session)
4. [Prepare a personal working branch](#4-prepare-a-personal-working-branch)
5. [Update one dataset](#5-standard-procedure-for-updating-one-dataset)

## 1. Install the Davis CLI

### Requirements

- Git
- GitHub access to the Davis repository
- The shared organizer code distributed through an organizer-only channel

Routine organizer operations do not require Rust, Cargo, Python, DVC, Google Drive, or R2 credentials. Rust and Cargo are required only when developing or building Davis itself.

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

This command downloads the Davis executable from GitHub Releases and places it where the `davis` command can run. It does not install Rust or Cargo.

Open a new terminal and verify the installation:

```text
davis --version
davis operator --help
```

## 2. Prepare the repository

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

Each command has the following purpose:

- `git status`: Displays the current branch and uncommitted changes without modifying files. `working tree clean` means there are no uncommitted changes.
- `git switch main`: Changes the working directory to `main`, the base of the official history.
- `git pull --ff-only`: Downloads the latest `main` from GitHub and advances the local branch only when the history remains a straight line. If local and remote history have diverged, it stops without merging. This is a fast-forward.

### Preparing the repository in VS Code

For a first clone, run `Git: Clone` from the Command Palette, enter the repository URL, choose a destination, and open the cloned folder. If the repository is already cloned, open its `davis` folder in VS Code.

Select the branch name in the lower-left Status Bar and switch to `main`. Then open the Source Control view, select `…`, and choose `Pull`. If changed files appear or VS Code requests a conflict resolution or merge, stop and consult the organizer team. The terminal form `git pull --ff-only` provides the strict guarantee that divergence stops without modifying history.

## 3. Prepare an organizer session

Specify the Davis Web URL provided by the organizer team and enter the shared organizer code at the interactive prompt. The code does not need to appear in a command argument or shell history.

```text
davis operator login <Davis Web URL>
davis operator status
```

Davis does not store the shared code itself. It stores only a revocable organizer session, so authentication normally does not need to be repeated while that session remains valid. `push` and `publish` use the organizer session, while `get` and `pull` require a separate download-only participant session. An organizer who retrieves real data should also run `davis login <Davis Web URL>` once.

In VS Code, open a terminal from the Terminal menu and enter the same commands. The session is stored in the PC's user configuration directory rather than in VS Code, so it is shared between a regular terminal and the VS Code terminal.

## 4. Prepare a personal working branch

Organizers update data on a personal working branch rather than `main`. Davis does not require a branch prefix or a match with a GitHub username; use any name your team can identify. Reusing one personal branch is recommended, but creating a new branch is also supported.

### First-time setup

Create the personal working branch from current `main`.

```bash
git status
git switch main
git pull --ff-only
git switch -c <personal-working-branch>
```

- `git switch -c <name>`: Creates a branch with the same contents as the current `main` and switches to it.
- A separate `git push` is not required here. The first `davis push` publishes the branch to GitHub.

In VS Code, select the branch name in the lower-left Status Bar, switch to `main`, and choose `Pull` from the Source Control view's `…` menu. Select the branch name again, choose `Create new branch...`, and enter any identifiable name. If VS Code asks for the source, select `main`.

### Before every later edit

Confirm that the previous Pull Request has been merged into `main`, then fast-forward the personal branch to current `main`.

```bash
git status
git switch <personal-working-branch>
git fetch origin
git merge --ff-only origin/main
```

- `git fetch origin`: Downloads current GitHub history without changing the current branch or working files.
- `git merge --ff-only origin/main`: Advances the personal branch to GitHub's current `main`. If history has diverged, it stops without creating a merge commit.

In VS Code, switch to the personal branch from the lower-left branch indicator and choose `Fetch` from the Source Control view's `…` menu. You can then run `Git: Merge Branch...` from the Command Palette and select `origin/main`. However, the normal VS Code Merge action does not explicitly enforce `--ff-only`. Use it only when the Source Control Graph confirms a straight fast-forward path. If VS Code shows conflicts, a merge commit, or uncommitted changes, do not complete the operation. The terminal command above is the reliable option.

`davis push` accepts any named branch other than `main`. It rejects `main` and a detached HEAD but does not prescribe a name format.

## 5. Standard procedure for updating one dataset

1. Before editing, switch to the personal branch and fast-forward it to the current `main` on GitHub.

   ```bash
   git status
   git switch <personal-working-branch>
   git fetch origin
   git merge --ff-only origin/main
   ```

   Confirm with `git status` that there are no uncommitted changes before continuing. If the final command fails, the histories have diverged. Do not start editing; consult the operations team.
2. If the published real data is not already available locally, retrieve it before editing. `pull` obtains or updates real data to the currently published Manifest. Do not run it after editing because it can overwrite local changes.

```bash
davis pull routes/Matsuyama
```

3. To confirm the starting data against the current Manifest, run `verify` before editing. It only inspects files. After the data is edited, failure is expected because the old Manifest no longer matches.

```bash
davis verify routes/Matsuyama
```

4. Edit the assigned dataset's real data and `schema.yaml`. Do not edit PDFs or DatasetManifests manually.
5. Inspect the planned update without changing anything. `--dry-run` hashes new and changed files and reports missing objects and upload bytes without changing the cache, PDFs, R2, or Git.

```bash
davis push routes/Matsuyama --dry-run
```

6. Check `Missing objects`, `Existing objects`, and `Upload size`. Do not continue if the files or size are unexpected.
7. Run the real push. `-m` supplies the description recorded in Git; omitting it uses `data: update <dataset>`.

```bash
davis push routes/Matsuyama -m "data: update routes/Matsuyama"
```

This single command creates Objects and the Manifest, uploads missing Objects to R2, generates Japanese and English PDFs, commits the selected files, and pushes the current personal branch to GitHub. Do not stage, commit, or push again in VS Code. Completion is reflected by a clean Source Control view with no outgoing commit in the lower-left sync indicator.

8. Open a Pull Request from the personal branch to `main` on GitHub and request review from another organizer. With the GitHub Pull Requests and Issues extension installed, VS Code can create it from `Create Pull Request` in the Source Control view. Otherwise, use GitHub Web.
9. After review, merge the Pull Request into `main` with a merge commit. Do not delete the personal branch if it will be reused.
10. Designate one publisher, switch to current `main`, and publish.

```bash
git switch main
git pull --ff-only
git status
davis publish
```

Here, `git status` confirms that publication starts from a clean working tree. `davis publish` independently checks the current `origin/main` and clean working tree and refuses publication if either condition fails. In VS Code, switch to `main` from the lower-left branch indicator and choose `Pull` from the Source Control view's `…` menu, then run `davis publish` in the VS Code terminal.

11. Confirm `Catalog published: yes`, force-refresh the Web catalog, and inspect names, schemas, licenses, file counts, PDFs, and downloads.

Routine work ends here. Read the remaining sections when you need detailed command roles, safety rationale, or exceptional-case recovery.

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

### Equivalent VS Code actions

| Git operation | VS Code action | Routine-work note |
| --- | --- | --- |
| Inspect or switch branches | Select the branch name in the lower-left Status Bar | Do not switch while files have pending changes |
| Create a branch | Select the branch name, then `Create new branch...` | Use current `main` as the source |
| Fetch | Source Control `…` → `Fetch` | Downloads history without changing files |
| Pull | Source Control `…` → `Pull` | Use it to update `main` |
| Push | Source Control `…` → `Push` | Normally unnecessary because `davis push` performs it |
| Pull and push | `Sync Changes` or the lower-left sync icon | Runs both operations, so it is not used in the standard Davis workflow |

VS Code Source Control actions and terminal Git commands modify the same repository state. An operation performed through either interface is visible in the other.

For current interface names and details, see the official VS Code documentation for [Branches and Worktrees](https://code.visualstudio.com/docs/sourcecontrol/branches-worktrees) and [Repositories and Remotes](https://code.visualstudio.com/docs/sourcecontrol/repos-remotes).

### Davis commands used by organizers

| Command | Purpose | Effect on the public Web catalog |
| --- | --- | --- |
| `davis --version` | Show the installed version | None |
| `davis update` | Check the latest release and update the CLI after approval | CLI binary |
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

The CLI checks for a newer release once every 24 hours and displays a notice after a normal command when an update is available. Run `davis update` to compare the installed version with the latest release and answer `y/N` before installation. Enter `y` to run the operating-system installer and verify the release artifact's SHA-256 checksum. Use `davis update --yes` to skip confirmation. The installer preserves the repository, real data, participant session, and organizer session.

## Using storage outside the official deployment

The personal-branch, `origin/main`, and Pull Request rules in this guide are the operating policy for safely maintaining the official Davis Catalog as a team. When no organizer session is active, `davis push` to a filesystem or S3-compatible remote configured in `.davis/config.toml` does not require the official branch name or GitHub and does not create or push a Git commit. Another organization can retain the common Object and Manifest formats while defining its own review and publication policy.

## Troubleshooting

- If `davis operator status` reports an expired session, run `davis operator login <URL>` again.
- If `davis` is not found, open a new terminal.
- If the legacy CLI still runs, use `which -a davis` on macOS or Linux, or `Get-Command davis -All` on Windows, to inspect executable paths.
- If migration fails while Git has uncommitted changes, do not force-reset the repository. Consult the organizer team.
