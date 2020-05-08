# Downloading Kinetics

Python script to download a CSV split:

```bash
$ python -m ar download_kinetics <kinteics_split.csv> \
    kinetics400/<train|validate|test> \
    -n 8 \ # Number of processes
    --failed-file logs.json # Not mandatory but recommended
```

For more information about the download script read it 
[here](ar/data/download.py).

To avoid downloading already existing files copy the whole `s3://actvnod/`
s3 bucket in your working directory.

