import os
import tempfile
import numpy as np
import skimage.io as io

from midap.segmentation.hybrid_segmentator import HybridSegmentation
from skimage.io import imread
from pytest import fixture
from pathlib import Path
from os import listdir

"""
Note:
Most methods tested here are actually from the abstract base class SegmentationPredictor. The interactive 
method of the weight selection is actually monkeypatched in.
"""

# Fixtures
##########


@fixture()
def img1():
    """
    Creates a test image
    :return: A test image used as base image
    """
    # define the images
    img = np.ones((16, 16))

    # we pad the image with 0s to have a bigger image
    pad_img = np.pad(
        img, pad_width=[[32, 32], [32, 32]], mode="constant", constant_values=0.0
    )

    return pad_img


@fixture()
def segmentation_instance(monkeypatch, img1):
    """
    This fixture prepares the HybridSegmentation class including monkey patching for the image read
    :param monkeypatch: The monkypatch fixture from pytest to override methods
    :param img1: A test image fixture used as base image to segment
    :return: A CutoutImage instance
    """

    # create a temp directory
    tmpdir = tempfile.TemporaryDirectory()

    # directories for the read and write
    path_cut = os.path.join(tmpdir.name, "cut_im")
    os.makedirs(path_cut)
    path_seg = os.path.join(tmpdir.name, "seg_im")
    os.makedirs(path_seg)
    path_seg = os.path.join(tmpdir.name, "seg_im_bin")
    os.makedirs(path_seg)

    def fake_list(directory):
        """
        Monkeypatch for os.listdir
        :param directory: path of the direcotry
        :return: A list of images depending on the input
        """

        # we return different images depending on the channel
        if "cut_im" in directory:
            return ["img1_cut.png", "img2_cut.png", "img3_cut.png"]

    # patch
    monkeypatch.setattr(os, "listdir", fake_list)

    def fake_load(path):
        """
        This is a monkeypatch for io.imread
        :param path: Path of the image to load
        :return: A loaded image
        """

        return img1

    # patch
    monkeypatch.setattr(io, "imread", fake_load)

    # get the instance
    unet = HybridSegmentation(
        data_type="Family_Machine",
        path_model_weights=[tmpdir.name], postprocessing=True, div=16, connectivity=1
    )

    yield unet

    # clean up
    tmpdir.cleanup()


# Tests
#######


def test_run_image_stack(segmentation_instance):
    """
    Tests the run_image_stack method of the SegmentationPredictor class and all other class functions in the process
    :param segmentation_instance: A pytest fixture preparing and monkeypatching the class instance for testing
    """

    # we read out the path model weights (because we stored the tempdir in there)
    channel_path = segmentation_instance.path_model_weights[0]

    # we set the segmentation method to watershed for starters
    segmentation_instance.model_weights = "watershed"

    # run the stack
    segmentation_instance.run_image_stack(channel_path=channel_path, clean_border=True)

    # we check that we got three files (we use the directly imported methods because the others were monkeypatched)
    seg_files = listdir(os.path.join(channel_path, "seg_im"))
    assert len(seg_files) == 3

    # we check that was labeled correctly
    for f in seg_files:
        fpath = os.path.join(channel_path, "seg_im", f)
        img = imread(fpath)
        # equal to 1 with new border removal
        assert np.unique(img).size == 1

    # path to actual model weights
    weight_path = Path(__file__).absolute().parent.parent.parent
    weight_path = weight_path.joinpath(
        "model_weights",
        "model_weights_family_mother_machine",
        "model_weights_CB15-WT.h5",
    )
    segmentation_instance.model_weights = str(weight_path)

    # run the stack
    segmentation_instance.run_image_stack(channel_path=channel_path, clean_border=True)

    # we check that was labeled correctly
    for f in seg_files:
        fpath = os.path.join(channel_path, "seg_im", f)
        img = imread(fpath)
        # equal to 1 with new border removal
        assert np.unique(img).size == 1
