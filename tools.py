import zipfile
from tqdm.auto import tqdm
import os
import shutil
import tarfile


# Use zipfile to unzip .zip files
def unzip(src, dst):
    f = zipfile.ZipFile(src, 'r')
    os.mkdir(dst)
    for file in tqdm(f.namelist()):
        f.extract(file, dst)
    f.close()


def move_dir(src, dst):
    shutil.move(src, dst)


def tar_package(src, dst):
    tar = tarfile.open(dst, 'w')
    tar.add(src, arcname='name.zip')
    tar.close()


def tar_extractor(src, dst):
    tar = tarfile.open(src, 'r')
    tar.extractall(dst)
    tar.close()


if __name__ == '__main__':
    src = "F:/6D Pose Estimation/scene1_low.zip"
    dst = "F:/6D Pose Estimation/lm/scene1_low"
    unzip(src, dst)

    # src = "./train_cqt_power"
    # dst = "F:/"
    # move_dir(src, dst)
    # src = "./test_cqt_power"
    # dst = "F:/"
    # move_dir(src, dst)

    # src = "./train_cqt_power"
    # dst = "F:/train_cqt_power.tar"
    # tar_package(src, dst)
    # src = "F:/train_cqt_power.tar"
    # dst = "F:/train_cqt_power"
    # tar_extractor(src, dst)
    #
    # src = "./test_cqt_power"
    # dst = "F:/test_cqt_power.tar"
    # tar_package(src, dst)
    # src = "F:/test_cqt_power.tar"
    # dst = "F:/test_cqt_power"
    # tar_extractor(src, dst)
