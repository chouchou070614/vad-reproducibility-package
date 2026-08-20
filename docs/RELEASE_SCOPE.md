# Anonymous submission artifact scope

## Included

- source code required to train and evaluate the reconstruction and prediction
  branches;
- the frozen memory-initialization, alignment, fusion, and temporal-refinement
  implementation used by the manuscript;
- compact numerical evidence needed to audit tables and rebuild the released
  qualitative and score-curve figures;
- exact paper configurations, environment specifications, tests, provenance,
  licensing notes, and a recursive SHA-256 manifest.

## Deliberately excluded

- datasets, video frames, full checkpoints, and memory tensors that cannot be
  redistributed or would exceed a compact review package;
- developer workspaces, absolute local or cluster paths, usernames, hostnames,
  scheduler files, credentials, and raw service logs;
- discarded exploratory mechanisms (including NCBAF-v2 and V3), historical
  duplicate exports, and label-selected parameter variants not reported as the
  frozen method;
- LaTeX sources, manuscript PDFs, reviewer correspondence, and author metadata;
- generated caches, editor settings, experiment trackers, and archives of the
  repository itself.

## Why some derived files remain

The PNG figures are retained alongside their compact CSV/NPZ inputs because
they let reviewers compare the released rendering against a deterministic
rebuild. The corresponding scripts are included, so these are verification
pairs rather than redundant historical copies.
