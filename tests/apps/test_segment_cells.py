from pathlib import Path
from shutil import copyfile

import numpy as np
import pytest
from pytest import mark
from skimage import io

from midap.apps.segment_cells import main


# Fixtures
##########


@pytest.fixture()
@mark.usefixtures("setup_dir")
def prep_dirs(setup_dir):
    """
    A fixture that copies the cut images into the temp directory for segmentation
    :param setup_dir: The path to the temp directory containing the setup and the channel
    :return: The path to the directory containing the channel directory, the channel name and the path containing
             the solution of the binary images
    """
    # unpack
    tmpdir_name, channel = setup_dir

    # copy all the files
    data_path = Path(__file__).parent.joinpath("data")
    src_files = data_path.joinpath("cut_im").glob("*.png")
    channel_dir = Path(tmpdir_name).joinpath(channel)
    dst_dir = channel_dir.joinpath("cut_im")
    for f in src_files:
        copyfile(src=f, dst=dst_dir.joinpath(f.name))

    return Path(tmpdir_name), channel, data_path.joinpath("seg_im_bin")


# Tests
#######


def test_main(prep_dirs):
    """
    Tests the main routine of the segment_cells app
    :param prep_dirs: The prep dirs fixtures which creates everything necessary to segment the cells on test images
    """

    # unpack
    path_pos, path_channel, sol_path = prep_dirs

    # prep the remaining arguments
    path_model_weights = [Path(__file__).parent.parent.parent.joinpath(
        "model_weights"
    )]
    postprocessing = True

    # Test for invalid segmentation class
    with pytest.raises(ValueError):
        main(
            data_type="MotherMachine",
            path_model_weights=path_model_weights,
            path_pos=path_pos,
            path_channel=path_channel,
            segmentation_class="This is not a valid class",
            postprocessing=postprocessing,
            clean_border=True,
        )

    # Tests for UNetSegmentation
    segmentation_class = "UNetSegmentation"
    network_name = path_model_weights[0].joinpath(
        "model_weights_legacy",
        "model_weights_C-crescentus-CB15_mKate2_v01.h5"
    )

    # just the selection
    network_name_new = main(
        data_type="Family_Machine",
        path_model_weights=path_model_weights,
        path_pos=path_pos,
        path_channel=path_channel,
        segmentation_class=segmentation_class,
        postprocessing=postprocessing,
        clean_border=True,
        network_name=network_name,
        just_select=True,
    )

    assert Path(network_name_new) == Path(network_name)

    # now actual segmentation
    _ = main(
        data_type="Family_Machine",
        path_model_weights=path_model_weights,
        path_pos=path_pos,
        path_channel=path_channel,
        segmentation_class=segmentation_class,
        postprocessing=postprocessing,
        clean_border=True,
        network_name=network_name,
        just_select=False,
    )

    # get the segmented images and the solution
    sol_imgs = sorted(sol_path.glob("*.png"))
    seg_imgs = sorted(path_pos.joinpath(path_channel, "seg_im_bin").glob("*.png"))

    # compare length
    assert len(seg_imgs) == len(sol_imgs)

    for sol_f, seg_f in zip(sol_imgs, seg_imgs):
        sol_img = io.imread(sol_f)
        seg_img = io.imread(seg_f)
        assert np.allclose(sol_img, seg_img)

    # Tests for Omni if supported
    try:
        import omnipose
        from cellpose import models

        segmentation_class = "OmniSegmentation"
        network_name = "bact_fluor_omni"

        # just the selection, testing the actual segmentation would require larger images
        network_name_new = main(
            data_type="Family_Machine",
            path_model_weights=path_model_weights,
            path_pos=path_pos,
            path_channel=path_channel,
            segmentation_class=segmentation_class,
            postprocessing=postprocessing,
            clean_border=True,
            network_name=network_name,
            just_select=True,
        )

        assert network_name_new == network_name
    except ImportError:
        pass
