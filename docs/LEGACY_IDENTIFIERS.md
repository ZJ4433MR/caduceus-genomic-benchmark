# Legacy Identifiers

The public model name is **Flipped-GARI**.

Several Python symbols and checkpoint state dictionaries predate that name and
therefore retain `MLBN` or `mlbn`:

- `MLBN_encoder`
- `DNAEmbeddingModelMLBN`
- `MLBNLMHeadModel`

These names are internal compatibility identifiers, not a second model. The
release exposes the following public names:

- `FlippedGARIEncoder`
- `DNAEmbeddingModelFlippedGARI`
- `FlippedGARILMHeadModel`
- `dna_embedding_flipped_gari`
- `flipped_gari_lm`
- `flipped-gari-local`
- `--flipped_gari_config`
- `--flipped_gari_checkpoint`

The aliases instantiate exactly the same implementation and do not change
the module hierarchy, state-dict keys, or model behavior.
