# Davis repository metadata

`datasets/` contains the versioned `DatasetManifest` files used by Davis.
The operator-facing `davis push [dataset]` command scans the selected dataset
roots directly and computes BLAKE3 object IDs. Omitting the dataset selects all
datasets. After missing immutable objects upload successfully, Davis regenerates
only affected schema PDFs, then commits and pushes the selected metadata on the
current personal `operator/<GitHub-username>` branch. DVC metadata is not used.

The separate `ingest` command remains available for offline diagnostics.
A normal push reuses unchanged files whose previous Manifest entry and local
cache object remain valid, and hashes new, changed, or uncached files. Use
`--rehash` only when every selected source must be read again.

The personal-branch and GitHub checks apply only to the official operator
session workflow. Direct filesystem and S3-compatible remotes remain usable
without the official repository or GitHub.

The content-addressed cache under `cache/` is generated locally and excluded
from Git. Storage credentials must not be written to this directory or to a
tracked configuration file. Copy `config.example.toml` to `config.toml`, fill
in only the non-secret remote settings, and provide S3-compatible credentials
at runtime.
