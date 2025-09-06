# Davis CLI: Dataset Management Tool<!-- omit in toc -->

<table>
    <thead>
        <tr>
            <th style="text-align:center">English</th>
            <th style="text-align:center"><a href="README.md">日本語</a></th>
        </tr>
    </thead>
</table>

`davis-cli` is a command-line tool for easily obtaining and managing datasets used in the Summer School for Behavioral Modeling.

> [!WARNING]
> Currently available only to participants of the Summer School for Behavioral Modeling.

## Table of Contents <!-- omit in toc -->

- [1. Main Features](#1-main-features)
- [2. Installation](#2-installation)
    - [2.1. Prerequisites](#21-prerequisites)
    - [2.2. Installation Procedure](#22-installation-procedure)
- [3. Initial Setup](#3-initial-setup)
    - [3.1. Creating a Google Cloud Project](#31-creating-a-google-cloud-project)
    - [3.2. Enabling the Google Drive API](#32-enabling-the-google-drive-api)
    - [3.3. Issuing an OAuth 2.0 Client ID](#33-issuing-an-oauth-20-client-id)
    - [3.4. Adding a Test User](#34-adding-a-test-user)
- [4. Using Datasets](#4-using-datasets)
    - [4.1. Listing Available Datasets](#41-listing-available-datasets)
    - [4.2. Viewing Detailed Information About a Dataset](#42-viewing-detailed-information-about-a-dataset)
    - [4.3. Downloading Datasets](#43-downloading-datasets)
- [5. Other Commands](#5-other-commands)
    - [5.1. Displaying Help](#51-displaying-help)
- [6. Support](#6-support)

## 1. Main Features

- List available datasets
- View detailed information about a dataset
- Download a specified dataset

## 2. Installation

### 2.1. Prerequisites

Git must be installed.
Install Git using the appropriate link for your operating system:

- **Windows**: [Git Official Site](https://git-scm.com/download/win)
- **macOS**: [Git Official Site](https://git-scm.com/download/mac)
- **Linux**: Install using your distribution's package manager (e.g., `sudo apt install git`).

Additionally, depending on your environment, you may need to manually install [DVC (Data Version Control)](https://dvc.org/).
If you encounter an error message stating DVC cannot be found when running the `davis get` command after completing the “Installation Steps” and “Initial Setup” below, please install DVC.

- **Installing DVC**: Install following the instructions on the [DVC official website](https://dvc.org/doc/install).
    - You can install it using `uv tool install dvc[gdrive]`.

### 2.2. Installation Procedure

Currently, `davis-cli` must be installed directly from the Git repository using either `uv` or `pip`.
(Release on PyPI is not planned.)
Unless you have a specific reason, we strongly recommend using `uv`.

```bash
# Using uv (strongly recommended. Use this unless you have a specific reason)
uv tool install git+https://github.com/bin-utokyo/davis#subdirectory=packages/dataset_cli

# Using pip
pip install git+https://github.com/bin-utokyo/davis.git#subdirectory=packages/dataset_cli
```

After installation, verify that it is in your PATH.
If the `davis --help` command executes successfully, installation is complete.

If you installed using `uv`, instructions for adding it to your PATH may be displayed; follow those instructions.

## 3. Initial Setup

After installation, first run the `config lang en` command to set the language to English (default is Japanese) if you prefer English.

```bash
davis config lang en
```

Then, run the `setup` command to configure the authentication credentials required for data download.

```bash
davis setup
```

Running this command will prompt you for the following three pieces of information:

1. **Google Drive Folder ID**
    - Enter the folder ID provided by the administrator.
2. **Google Cloud Client ID**
3. **Google Cloud Client Secret**

This information is required to access the Google Drive where data is stored.

You must obtain the Google Cloud Client ID and Client Secret yourself from the Google Cloud Console.
Detailed instructions for obtaining them are provided below.
Note that some of the screen text may differ.

> [!NOTE]
> This operation will not incur any charges.
> Additionally, the administrator cannot access your Google account information.
> Please note that the administrator cannot be held responsible for any issues arising from this operation.

### 3.1. Creating a Google Cloud Project

1. Access the [Google Cloud Console](https://console.cloud.google.com/) and log in with your Google account.
2. Click “New project” from the project selection menu at the top of the screen and create a new project with any name (e.g., `davis-data-access`).

### 3.2. Enabling the Google Drive API

1. Ensure the newly created project is selected. From the left navigation menu, select “APIs and Services” > “Library”.
2. Enter “**Google Drive API**” in the search bar and click the displayed API.
3. Click the “Enable” button to activate the API.

### 3.3. Issuing an OAuth 2.0 Client ID

1. From the left navigation menu, select “APIs & Services” > “Credentials”.
2. Click “Create credentials” at the top of the screen and select “OAuth client ID”.
3. If prompted to “Configure consent screen”, set it up using the following steps. Your email address will not be publicly displayed.
    - **App name**: Enter a descriptive name, such as `davis-cli`.
    - **User support email**: Select your own email address.
    - **User type**: Select “External”.
    - **Contact Information**: Enter your email address.
    - Agree to the policy, click “Next”, then click “Create”.
4. Return to the “Credentials” screen, select “Create credentials” > “OAuth client ID”.
5. Under “Application type,” select “**Desktop app**.”
6. Enter a name like “davis-cli-local” and click “Create.”
7. Once created, the “**Client ID**” and “**Client Secret**” will be displayed. Copy these and paste them into the prompt for the `davis setup` command.

Once all information is entered, the configuration is saved to `.config/davis/config.ini` in your home directory, and the tool is ready.

### 3.4. Adding a Test User

1. From the left navigation menu, select “API and Services” > “OAuth Consent Screen”.
2. Click “Audience” on the left and scroll to the “Test Users” section.
3. Click “Add users” and add your Google account email address.

## 4. Using Datasets

### 4.1. Listing Available Datasets

Use the `list` command to view a list of downloadable datasets.

```bash
davis list
```

### 4.2. Viewing Detailed Information About a Dataset

Use the `info` command to view information about the files and documents contained in a specific dataset.

```bash
davis info <DATASET_ID>
```

- Replace `<DATASET_ID>` with the ID displayed by the `list` command.
- Adding the `--open` option opens the associated documentation (PDF) in your browser. This document contains detailed descriptions of the dataset, including column types.

```bash
# Example: Display information for the PT_data dataset and open the PDF in the browser
davis info PT_data --open
```

### 4.3. Downloading Datasets

Use the `get` command to download datasets locally.

You can download files individually or download entire directories at once.

```bash
davis get <DATASET_ID>
```

- Use the `--out` or `-o` option to specify the download destination directory.

```bash
# Example: Download the PT_data dataset to the ‘my_data’ directory
davis get PT_data -o my_data/
```

> [!NOTE]
> First-time execution requires Google account authentication. Your browser will launch automatically and prompt you for authentication.
> You may see a warning stating “This app is not verified by Google,” but this is normal. Click the “Continue” link at the bottom of the screen and select “Go to davis-cli (unsafe page)” to proceed.

## 5. Other Commands

### 5.1. Displaying Help

Detailed usage for each command can be checked using the `--help` option.

```bash
davis --help
davis list --help
davis info --help
davis get --help
```

Similarly, you can check details for other commands using the `--help` option.

Note that the `davis setup` command only needs to be run once. Only re-run it if you wish to update your credentials.

Also, the `davis manage` command is intended for dataset administrators and is not necessary for regular users.

## 6. Support

If you encounter any issues or have questions, please contact the organizers via the Summer School for Behavioral Modeling Slack.
