[CmdletBinding()]
param(
    [string]$Version = $(if ($env:DAVIS_VERSION) { $env:DAVIS_VERSION } else { "latest" }),
    [string]$InstallDir = $(if ($env:DAVIS_INSTALL_DIR) { $env:DAVIS_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA "Davis\bin" })
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repositoryUrl = if ($env:DAVIS_REPOSITORY_URL) {
    $env:DAVIS_REPOSITORY_URL.TrimEnd("/")
} else {
    "https://github.com/bin-utokyo/davis"
}

$architecture = switch ($env:PROCESSOR_ARCHITECTURE.ToUpperInvariant()) {
    "AMD64" { "x86_64" }
    "ARM64" { "aarch64" }
    default { throw "Unsupported CPU architecture: $env:PROCESSOR_ARCHITECTURE" }
}
$asset = "davis-windows-$architecture.zip"
$downloadRoot = if ($Version -eq "latest") {
    "$repositoryUrl/releases/latest/download"
} else {
    "$repositoryUrl/releases/download/$Version"
}

$temporaryDirectory = Join-Path ([System.IO.Path]::GetTempPath()) ("davis-install-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $temporaryDirectory | Out-Null

try {
    $archivePath = Join-Path $temporaryDirectory $asset
    $checksumsPath = Join-Path $temporaryDirectory "SHA256SUMS"
    Write-Host "Downloading $asset..."
    Invoke-WebRequest -UseBasicParsing -Uri "$downloadRoot/$asset" -OutFile $archivePath
    Invoke-WebRequest -UseBasicParsing -Uri "$downloadRoot/SHA256SUMS" -OutFile $checksumsPath

    $escapedAsset = [regex]::Escape($asset)
    $checksumLine = Get-Content $checksumsPath | Where-Object {
        $_ -match "^([0-9a-fA-F]{64})\s+\*?$escapedAsset$"
    } | Select-Object -First 1
    if (-not $checksumLine) {
        throw "Checksum for $asset was not found"
    }
    $expectedChecksum = ([regex]::Match($checksumLine, "^[0-9a-fA-F]{64}")).Value.ToLowerInvariant()
    $actualChecksum = (Get-FileHash -Algorithm SHA256 -Path $archivePath).Hash.ToLowerInvariant()
    if ($actualChecksum -ne $expectedChecksum) {
        throw "Checksum verification failed"
    }

    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($env:DAVIS_SKIP_LEGACY_UNINSTALL -ne "1" -and $uv) {
        $uvTools = (& uv tool list 2>$null) -join "`n"
        if ($uvTools -match "(?m)^davis-cli\s") {
            Write-Host "Removing the legacy Python Davis CLI..."
            & uv tool uninstall davis-cli | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to remove the legacy Davis CLI"
            }
        }
    }

    $expandedDirectory = Join-Path $temporaryDirectory "expanded"
    Expand-Archive -Path $archivePath -DestinationPath $expandedDirectory -Force
    $source = Join-Path $expandedDirectory "davis.exe"
    if (-not (Test-Path $source -PathType Leaf)) {
        throw "The release archive does not contain davis.exe"
    }

    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    $destination = Join-Path $InstallDir "davis.exe"
    Copy-Item -Path $source -Destination $destination -Force

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $pathEntries = @($userPath -split ";" | Where-Object { $_ })
    if (-not ($pathEntries | Where-Object { $_.TrimEnd("\") -ieq $InstallDir.TrimEnd("\") })) {
        $newUserPath = if ($userPath) { "$InstallDir;$userPath" } else { $InstallDir }
        [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
        Write-Host "Added $InstallDir to the user PATH."
    }
    $env:Path = "$InstallDir;$env:Path"

    & $destination --version
    Write-Host "Davis was installed to $destination."
    Write-Host "Open a new terminal before using the davis command."
} finally {
    Remove-Item -Recurse -Force $temporaryDirectory -ErrorAction SilentlyContinue
}
