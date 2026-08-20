# Davis repository metadata

`datasets/` contains versioned `DatasetManifest` files generated from the
current legacy DVC metadata and verified local data.

The content-addressed cache under `cache/` is generated locally and excluded
from Git. Storage credentials must not be written to this directory or to a
tracked configuration file. Copy `config.example.toml` to `config.toml`, fill
in only the non-secret remote settings, and provide S3-compatible credentials
at runtime.
