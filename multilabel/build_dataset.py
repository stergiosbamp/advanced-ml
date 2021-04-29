import glob
import os
import PIL
import shutil
import tarfile
import urllib
import zipfile

import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer


def build_dataset(urls, dest_dir="data/chest_x_rays", target_size=(299,299)):

    # if dest_dir doesn't exist, make it
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for i, url in enumerate(urls):
        # set a filename for the downlaod
        fn = f"images_{i + 1:02d}.tar.gz"

        # download compressed file
        print(f"downloading {fn}")
        urllib.request.urlretrieve(url, fn)

        # extract file
        print(f"extracting {fn}")
        extract_dir = fn.split(".")[0]
        with tarfile.open(fn) as f:
            f.extractall(extract_dir)

        # delete compressed file
        print(f"removing {fn}")
        os.remove(fn)

        # get image files
        images = glob.glob(extract_dir + "/images/*.png")

        # resize images
        print("resizing images")
        for image in images:
            PIL.Image.open(image).resize(size=target_size).save(image)

        # move images in same directory
        print("moving images")
        for image in images:
            dest_image = dest_dir + "/" + image.split("/")[-1]
            shutil.move(image, dest_image)

        # delete unused dir
        print("removing unused dir")
        shutil.rmtree(extract_dir)


def zip_dataset(output_filename, dataset_dir="data/chest_x_ray"):
    
    # zip dataset in external folder (e.g. Google Drive)
    shutil.make_archive(output_filename, "zip", dataset_dir)


def build_dataset_from_zip(input_file, dest_dir="data/chest_x_ray"):

    # build dataset from external zip file (e.g. Google Drive)
    with zipfile.ZipFile(input_file) as f:
        f.extractall(dest_dir)


def build_binarized_annotations(
    data_entry_csv="data/Data_Entry_2017_v2020.csv",
    dest_dir="data"):
    
    # read data entry
    df = pd.read_csv(data_entry_csv)

    # keep only filenames and labels
    df = df[["Image Index", "Finding Labels"]]

    # rename columns
    df = df.rename(columns={"Image Index": "filename", "Finding Labels": "labels"})

    # define function to parse label string
    def parse_label(label_string):
        parsed = label_string.split("|")
        if parsed == ["No Finding"]:
            parsed = []
        return parsed
    
    # parse labels
    df["labels"] = df["labels"].apply(parse_label)

    # fit MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit(df["labels"])

    # transform labels
    df["binarized_labels"] = df["labels"].apply(lambda x: mlb.transform([x])[0])

    # save csv with image paths and annotations
    df.to_csv(dest_dir + "/" "binarized_annotations.csv")

    # save csv with class mapping
    class_mapping = pd.Series(mlb.classes_)
    class_mapping.to_csv(dest_dir + "/" + "class_mapping.csv", header=False)


if __name__ == "__main__":

    # URLs for the zip files
    urls = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
    	'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
    	'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
    	'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
    	'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
    	'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
    	'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
    	'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]

    #build_dataset(urls)
    build_binarized_annotations()
