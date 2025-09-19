import boto3
import os
import pandas as pd
from tqdm import tqdm
from mhd_utils import load_mhd_u8_from_s3
from s3_utils import save_npy_to_s3, append_df_to_s3_csv

BUCKET = "vessel-segmentation-data"
PREFIX = "data/raw/SintefData"


def main():

    s3 = boto3.client("s3")

    # set up paginator to get all objects under PREFIX
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)

    # set up dict to store data
    df_dict = {
        'frame_path': [],
        'segm_path': [],
        'model_dataset': [],
        'source_dataset': [],
        'patient_id': [],
        'raw_path': []
    }

    # loop through and find all .mhd files under PREFIX
    for page in page_iterator:
        for obj in tqdm(page.get("Contents", [])):
            key = obj["Key"]

            # segmentations have same name as frames except they end in _gt
            filename = key.split('/')[-1]
            basename = filename.split('.')[0]
            if key.lower().endswith(".mhd") and basename.endswith("_gt"):

                # get frame .mhd path
                dirname = '/'.join(key.split('/')[:-1])
                frame_mhd = os.path.join(
                    dirname,
                    basename[:-3] + '.mhd'
                )

                # form new paths
                frame_path = os.path.join(
                    'data/processed/SintefData/images',
                    '_'.join(dirname.split('/')[3:])
                ) + '_' + basename[:-3] + '.npy'
                segm_path = os.path.join(
                    'data/processed/SintefData/labels',
                    '_'.join(dirname.split('/')[3:])
                ) + '_' + basename[:-3] + '.npy'

                # Patient folder = immediate child under SintefData
                patient_id = key.split("/")[3]

                # for Sintef data
                #   patients 1-7 -> train
                #   patient 8 -> val
                #   patient 9-10 -> test
                if int(patient_id) <= 7:
                    model_ds = 'train'
                elif int(patient_id) == 8:
                    model_ds = 'val'
                else:
                    model_ds = 'test'

                # add to dict
                df_dict['frame_path'].append(frame_path)
                df_dict['segm_path'].append(segm_path)
                df_dict['model_dataset'].append(model_ds)
                df_dict['source_dataset'].append('SintefData')
                df_dict['patient_id'].append(patient_id)
                df_dict['raw_path'].append(frame_mhd)

                # read the numpy arrays and save them to s3
                frame_npy = load_mhd_u8_from_s3(BUCKET, frame_mhd)
                save_npy_to_s3(frame_npy, BUCKET, frame_path)
                segm_npy = load_mhd_u8_from_s3(BUCKET, key)
                save_npy_to_s3(segm_npy, BUCKET, segm_path)

    # save DataFrame to s3
    df = pd.DataFrame(df_dict)
    append_df_to_s3_csv(df, BUCKET, 'data/processed/manifest.csv')


if __name__ == '__main__' :
    main()