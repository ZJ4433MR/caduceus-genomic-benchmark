# Legacy Identifiers

The public model name is **Flipped-GARI**.

Several Python symbols and checkpoint state dictionaries predate that name and
therefore retain `MLBN` or `mlbn`:

- `MLBN_encoder`
- `DNAEmbeddingModelMLBN`
- `MLBNLMHeadModel`
- legacy registry keys `dna_embedding_mlbn` and `mlbn_lm`
- compatibility aliases `--mlbn_config` and `--mlbn_checkpoint` for the public
  VEP options `--flipped_gari_config` and `--flipped_gari_checkpoint`

Renaming these serialized keys would break strict checkpoint provenance. The
package adds public aliases:

- `dna_embedding_flipped_gari`
- `flipped_gari_lm`
- `flipped-gari-local`

The aliases instantiate exactly the same implementation and do not change
model behavior.
