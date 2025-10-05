# Data Handling

The functions in this folder are used for storing, processing, and organizing the ultrasound data for training.
If you would like to add a new data set, store it on the S3 Bucket `vessel-segmentation-data` under `data/raw` in its own folder.
Please add a description to [Data Sources](#data-sources) below explaining where the data comes from.
Then, since our data is stored in different formats, you will need to run some functions to process it into a consistent format and save it under `data/processed`.
See the [S3 Bucket description](s3-bucket-description) below for more details.

## S3 Bucket Description

The ultrasound data is stored in an s3 bucket called `vessel-segmentation-data`.

In this bucket, the folder `data/raw` contains the raw data from different sources, separated by subfolders.

The folder `data/processed` contains train, val, and test folders, each of which have exactly two subfolders: `images` and `labels`.
These folders contain .npy files with the same name, both uint8 and of the same shape (the labels are segmentations with 1 marking vessel pixels and 0 marking non-vessel pixels).

*Importantly*, there is a csv object `data/processed/manifest.csv` which contains the following columns describing every data point which has been processed.
- **frame_path**: the exact location of the .npy file containing the image
- **segm_path**: the exact location of the .npy file containing the segmentation
- **source_dataset**: which data set this example came from
- **patient_id**: a unique patient identifier *within the source dataset*
- **raw_path**: the exact location of the original file in the raw data

## Data Sources

- `SintefData`: this is the initial data which Sintef trained their model on. It contains imagery from 10 patients.
- `Paraguay`: TO BE ADDED!

## File Descriptions

- `s3_utils.py` contains helper functions for writing .npy files and updating .csv files on the s3 bucket
- `sanity_check.py` can be run periodically to check if all the .npy files in `data/processed/manifest.csv` actually exist; this file also double checks that every instance from the same patient falls in the same train, val, or test data set
- `mhd_utils.py` contains helper functions for reading the .mhd files which Sintef created in their original training
- `process_sintef_data.py` processes the data in `data/raw/SintefData`