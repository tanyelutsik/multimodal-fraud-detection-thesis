# Dataset Design

## Goal
This thesis compares tabular, image, and multimodal fraud detection.

## Tabular dataset
- Dataset: PaySim
- Task: binary fraud detection
- Target label: isFraud

## Image datasets
- MIDV: bona fide document images
- FantasyID: bona fide and manipulated/fake document images
- FMIDV: forged/manipulated document images

## Final image label mapping
- MIDV: 0 -> bona_fide
- FantasyID: bonafide -> bona_fide
- FantasyID: attack -> forged
- FMIDV: forged -> forged

## Final multimodal labels
- bona_fide
- forged

## Pairing strategy
Synthetic pairing was used.
Samples were paired:
- only inside the same split
- only inside the same class
- randomly

This means:
- train with train
- validation with validation
- test with test

## Why splits were not mixed
The original split boundaries were preserved to keep evaluation cleaner and avoid leakage.

## Limitation
The multimodal pairs are synthetic, so the tabular and image sample do not come from the same real event/person.

## Next steps
- verify final processed datasets and split sizes
- confirm label counts for tabular, image, and multimodal data
- freeze the final dataset methodology
- train the tabular baseline
- train the visual baseline
- train the multimodal model
- save metrics, confusion matrices, and notes for thesis writing