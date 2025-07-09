import gzip
import shutil

with gzip.open('experiment_00001.gz', 'rb') as f_in:
    with open('filename', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
