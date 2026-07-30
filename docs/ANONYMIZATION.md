# Anonymization

The AAAI reviewer artifact is built as a ZIP, not as a link to a source-control
repository.

`scripts/build_anonymous_archive.py` excludes:

- `.git` and other version-control metadata;
- raw datasets, model checkpoints, caches, and generated outputs;
- local build/staging directories;
- files containing common access-token patterns;
- files containing author names, local user profiles, cluster usernames, or
  machine-specific absolute paths.

Official third-party dataset URLs and bibliographic identifiers remain in the
archive because they identify external data sources rather than the authors.

Before upload:

```bash
python scripts/validate_package.py
python scripts/build_anonymous_archive.py
```

Inspect the resulting ZIP listing and the included
`PACKAGE_MANIFEST.sha256`. Do not add a private repository URL, author contact,
institution, acknowledgments, or non-anonymous checkpoint hosting location.
