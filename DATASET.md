# Dataset

## IAM Handwriting Database — Word Level

| Property | Details |
|----------|---------|
| Name | IAM Handwriting Database (IAM_Words) |
| Source | University of Bern, FKI Research Group |
| Size | ~115,320 word images (≈12,500 usable after filtering) |
| Format | PNG grayscale images + `words.txt` annotation file |
| License | Free for non-commercial research use |

### Automatic Download

The dataset is downloaded **automatically** when you run training:

```bash
python run.py train
```

The script downloads and extracts the dataset to `App/handwriting_recognition/Datasets/IAM_Words/`.

### Manual Download

If automatic download fails, download manually from the hosted mirror:

```
https://git.io/J0fjL
```

Extract the ZIP to `App/handwriting_recognition/Datasets/IAM_Words/`, then extract the inner archive:

```bash
cd App/handwriting_recognition/Datasets/IAM_Words
tar -xzf words.tgz -C words/
```

### Annotation Format

Each non-comment line in `words.txt` follows:

```
word_id status graylevel components x y w h tag transcription
```

- `word_id` — unique ID used to derive the image file path
- `status` — `ok` (valid) or `err` (segmentation error, skipped)
- `transcription` — ground-truth text label

### Citation

Marti, U.-V. and Bunke, H. (2002).
*The IAM-database: an English sentence database for offline handwriting recognition.*
International Journal on Document Analysis and Recognition (IJDAR), 5(1), pp. 39–46.
