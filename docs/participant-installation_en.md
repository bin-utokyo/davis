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
```

`get` recreates the `data/...` hierarchy below the directory where you run the command. You can also select files in Davis Web and use “Copy CLI command” to generate the corresponding command.

## Updating

The CLI checks for a newer release once every 24 hours and displays a notice after a normal command when an update is available. A failed update check never interrupts catalog searches or downloads.

```text
davis update
```

This command compares the installed version with the latest release and displays the update command for the current operating system. Run the displayed installer to update. The installer preserves the login session and downloaded object cache.

## Troubleshooting

- If `davis` is not found, open a new terminal.
- If the legacy CLI still runs, use `which -a davis` on macOS or Linux, or `Get-Command davis -All` on Windows, to inspect the executable paths.
- If the invite code is rejected, confirm that you are using the participant code and the URL supplied by the organizers.
- Do not enter the organizer code at the participant login prompt. The two codes are managed separately.
