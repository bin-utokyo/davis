#!/bin/sh

set -eu

REPOSITORY_URL="${DAVIS_REPOSITORY_URL:-https://github.com/bin-utokyo/davis}"
VERSION="${DAVIS_VERSION:-latest}"
INSTALL_DIR="${DAVIS_INSTALL_DIR:-$HOME/.local/bin}"

fail() {
    printf 'Davis installer: %s\n' "$1" >&2
    exit 1
}

command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v tar >/dev/null 2>&1 || fail "tar is required"

case "$(uname -s)" in
    Darwin) platform="macos" ;;
    Linux) platform="linux" ;;
    *) fail "this installer supports macOS and Linux; use install.ps1 on Windows" ;;
esac

case "$(uname -m)" in
    arm64|aarch64) architecture="aarch64" ;;
    x86_64|amd64) architecture="x86_64" ;;
    *) fail "unsupported CPU architecture: $(uname -m)" ;;
esac

asset="davis-${platform}-${architecture}.tar.gz"
if [ "$VERSION" = "latest" ]; then
    download_root="${REPOSITORY_URL}/releases/latest/download"
else
    download_root="${REPOSITORY_URL}/releases/download/${VERSION}"
fi

temporary_directory=$(mktemp -d "${TMPDIR:-/tmp}/davis-install.XXXXXX")
trap 'rm -rf "$temporary_directory"' EXIT HUP INT TERM

printf 'Downloading %s...\n' "$asset"
curl --proto '=https' --tlsv1.2 -fsSL "${download_root}/${asset}" \
    -o "${temporary_directory}/${asset}"
curl --proto '=https' --tlsv1.2 -fsSL "${download_root}/SHA256SUMS" \
    -o "${temporary_directory}/SHA256SUMS"

expected_checksum=$(awk -v filename="$asset" '$2 == filename || $2 == "*" filename { print $1 }' \
    "${temporary_directory}/SHA256SUMS")
[ -n "$expected_checksum" ] || fail "checksum for ${asset} was not found"
if command -v sha256sum >/dev/null 2>&1; then
    actual_checksum=$(sha256sum "${temporary_directory}/${asset}" | awk '{ print $1 }')
elif command -v shasum >/dev/null 2>&1; then
    actual_checksum=$(shasum -a 256 "${temporary_directory}/${asset}" | awk '{ print $1 }')
else
    fail "sha256sum or shasum is required"
fi
[ "$actual_checksum" = "$expected_checksum" ] || fail "checksum verification failed"

if [ "${DAVIS_SKIP_LEGACY_UNINSTALL:-0}" != "1" ] && command -v uv >/dev/null 2>&1 && uv tool list 2>/dev/null | grep -q '^davis-cli '; then
    printf 'Removing the legacy Python Davis CLI...\n'
    uv tool uninstall davis-cli >/dev/null || fail "failed to remove the legacy Davis CLI"
fi

mkdir -p "$INSTALL_DIR"
tar -xzf "${temporary_directory}/${asset}" -C "$temporary_directory"
[ -f "${temporary_directory}/davis" ] || fail "the release archive does not contain davis"
install -m 755 "${temporary_directory}/davis" "${INSTALL_DIR}/davis"

case ":$PATH:" in
    *":${INSTALL_DIR}:"*) ;;
    *)
        if [ "$platform" = "macos" ]; then
            profile="$HOME/.zshrc"
        else
            profile="$HOME/.profile"
        fi
        path_line='export PATH="$HOME/.local/bin:$PATH"'
        if [ "$INSTALL_DIR" = "$HOME/.local/bin" ] && ! grep -Fqx "$path_line" "$profile" 2>/dev/null; then
            printf '\n%s\n' "$path_line" >> "$profile"
            printf 'Added %s to PATH in %s.\n' "$INSTALL_DIR" "$profile"
        else
            printf 'Add %s to PATH if the davis command is not found.\n' "$INSTALL_DIR"
        fi
        ;;
esac

"${INSTALL_DIR}/davis" --version
printf 'Davis was installed to %s.\n' "${INSTALL_DIR}/davis"
printf 'Open a new terminal before using the davis command.\n'
