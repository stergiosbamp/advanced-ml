import glob
import os
import PIL
import shutil
import tarfile
import urllib


def build_dataset(urls, dest_dir="data/chest_x_rays", target_size=(299,299)):

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

    build_dataset(urls)
