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

## Participant login

Organizers who retrieve data through the CLI also sign in with the participant code:

```text
davis login https://davis-web.davis-bin.workers.dev
```

The participant session is download-only.

## Organizer login

```text
davis operator login https://davis-web.davis-bin.workers.dev
davis operator status
```

Enter the organizer-only shared code at `Operator code:`. Davis does not store the code itself. It stores a restricted organizer session that is valid for 30 days by default. You can push repeatedly without entering the code again during that period. After expiration, the next interactive `push` asks for the code once and renews the session.

## Checking connectivity

Run a dry run for the dataset you maintain from the repository root:

```text
davis push routes/Matsuyama --dry-run
```

The setup is ready when the output shows `Remote: ... (operator session)` together with the object counts and planned upload size. `--dry-run` does not modify R2 or the published catalog.

## Operating principles

- Update `main` before creating a personal working branch.
- You do not need to retrieve real data for datasets outside your assignment.
- Review schemas and metadata through Git, and send real data to R2 through Davis.
- A production `davis push` updates the Web catalog, so complete the dry run and review first.
- Never place the organizer code in the repository, an issue, a Pull Request, or a message with public recipients.
- If the code leaks, rotate the shared code and organizer access revision to invalidate all existing organizer sessions.

Until the DatasetManifest-first workflow is finalized, follow the organizer team's current update procedure for adding, moving, or removing data files. Do not run a production push only to test installation.

## Updating

The CLI checks for a newer release once every 24 hours and displays a notice after a normal command when an update is available. Run `davis update` to compare the installed version with the latest release and display the update command for the current operating system. Run the displayed installer to update. The installer preserves the repository, real data, participant session, and organizer session.

## Troubleshooting

- If `davis operator status` reports an expired session, run `davis operator login <URL>` again.
- If `davis` is not found, open a new terminal.
- If the legacy CLI still runs, use `which -a davis` on macOS or Linux, or `Get-Command davis -All` on Windows, to inspect executable paths.
- If migration fails while Git has uncommitted changes, do not force-reset the repository. Consult the organizer team.
