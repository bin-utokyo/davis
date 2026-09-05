# Davis Installation Guide for Participants

[日本語](participant-installation.md)

This guide is for participants who retrieve transport data through Davis Web or the Davis CLI. Organizers who publish data to R2 should use the [Installation Guide for Organizers](operator-installation_en.md).

## Using only Davis Web

You do not need to install an application when you only download data from Davis Web. Open the URL provided by the organizers in a browser and enter the shared participant code.

After signing in, you can inspect schemas, search and filter the catalog, select the required files, and download them.

## Using the CLI

The Davis CLI uses prebuilt executables for macOS, Windows, and Linux. You do not need Rust, Cargo, or Python. The installer detects the operating system and CPU architecture automatically and verifies the SHA-256 checksum of the downloaded file.

### macOS and Linux

Run the following command in a terminal:

```bash
curl --proto '=https' --tlsv1.2 -fsSL https://raw.githubusercontent.com/bin-utokyo/davis/main/scripts/install.sh | sh
```

### Windows

Run the following command in PowerShell:

```powershell
irm https://raw.githubusercontent.com/bin-utokyo/davis/main/scripts/install.ps1 | iex
```

If the legacy Python `davis-cli` was installed through `uv`, the installer removes it before installing the current CLI. It does not delete downloaded data, existing repository clones, or Davis login sessions.

## Verifying the installation

Open a new terminal or PowerShell window after the installer completes, then run:

```text
davis --version
davis --help
```

The installation is complete when `davis --help` lists `login`, `list`, `info`, and `get`.

## Adding Davis Desktop

After installing the CLI, you can add and launch Davis Desktop from any directory. You do not need to clone the GitHub repository or install Rust, Cargo, or `pnpm`.

```text
davis install desktop
davis desktop
```

Run `davis installed` to inspect the managed Desktop and model components. Desktop is stored in the Davis user data directory for the operating system, so launching it does not depend on the directory where it was installed.

## Participant login

Specify the Davis Web URL provided by the organizers:

```text
davis login https://davis-web.davis-bin.workers.dev
```

When `Invite code:` appears, enter the shared participant code. The input is hidden. You do not need to enter the code again while the session remains valid. Browser and CLI sessions are separate, so the CLI requires a one-time login even if you already signed in through the browser.

## Finding and retrieving data

```text
davis list
davis info routes/Matsuyama
davis get routes/Matsuyama
davis pull routes/Matsuyama
```

Use `get` for a first retrieval or selective file retrieval. Use `pull` either to retrieve a whole dataset for the first time or to synchronize an existing dataset to the current Manifest. Both commands recreate the `data/...` hierarchy below the directory where you run them. Davis Web uses `get` when you select files and choose “Copy CLI command.”

When files already exist at the `get` destination, Davis asks whether to replace them. Enter `y` to replace all selected existing files, or enter `N` (or press Enter) to cancel without changing them. Use `davis get <dataset> --force` to replace them without confirmation.

`pull` replaces existing files with the remote contents. If local files contain unfinished edits, preserve or finish that work before running it.

Running `davis pull` without a dataset ID retrieves or synchronizes every dataset. Specify an ID when only one dataset is needed.

## Updating

The CLI checks for a newer release once every 24 hours and displays a notice after a normal command when an update is available. A failed update check never interrupts catalog searches or downloads.

```text
davis update
```

This command retrieves update metadata directly from GitHub Releases, compares the installed version with the latest release, and asks `y/N` before installation. Enter `y` to run the operating-system installer automatically and verify the release artifact's SHA-256 checksum. Use `davis update --yes` to skip confirmation. The installer preserves the login session and downloaded object cache.

## Troubleshooting

- If `davis` is not found, open a new terminal.
- If the legacy CLI still runs, use `which -a davis` on macOS or Linux, or `Get-Command davis -All` on Windows, to inspect the executable paths.
- If the invite code is rejected, confirm that you are using the participant code and the URL supplied by the organizers.
- Do not enter the organizer code at the participant login prompt. The two codes are managed separately.
