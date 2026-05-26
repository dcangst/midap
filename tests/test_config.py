import os
import tempfile
from pathlib import Path

import pytest

from midap.config import Config


# Fixtures
##########


@pytest.fixture()
def tmp_dir():
    """
    Creates a tmp dir for a test and deletes it afterwards
    :return: The name of the directory as a string
    """

    # create
    tmpdir = tempfile.TemporaryDirectory()

    # the name
    yield tmpdir.name

    # clean up
    tmpdir.cleanup()


# Tests
#######


def test_Config(tmp_dir):
    """
    Tests the functionality of the Config class
    """

    # General settings and Family Machine
    # -----------------------------------

    # get the class
    config = Config(fname="settings.ini", general={"foo": "bar", "FolderPath": "DoesNotExist"})

    # check if we have the correct extra setting
    assert config.get("General", "foo") == "bar"
    assert config.get("General", "FolderPath") == "DoesNotExist"

    # the folder path shouldn't not exists
    with pytest.raises(FileNotFoundError):
        config.validate_general()

    # now we set it and the identifier should be the problem
    config.set("General", "FolderPath", tmp_dir)
    with pytest.raises(ValueError):
        config.validate_general()

    # now it should work
    config.set("General", "IdentifierFound", "pos1,pos2")
    config.validate_general()

    # we set an identifier
    config.set_id_section("pos1")
    config.set_id_section("pos2")

    # check all the id section tests
    config.validate_id_section("pos1", basic=True)

    # everything about corners
    with pytest.raises(ValueError):
        config.validate_id_section("pos1", basic=False)

    config.set("pos1", "Corners", "1,2,3,huhu")
    with pytest.raises(ValueError):
        config.validate_id_section("pos1", basic=False)
    config.set("pos1", "Corners", "1,2,3,4")

    # and the model weights
    try:
        # if omni pose is supported we test it
        import omnipose
        from cellpose import models

        config.set("pos1", "SegmentationClass", "OmniSegmentation")
        config.set("pos1", "ModelWeights_None", "bact_phase_omni")
    except ImportError:
        # otherwise we use the standard but add existing model weights
        path_model_weights = Path(__file__).parent.parent.joinpath(
            "model_weights",
            "model_weights_legacy",
            "model_weights_C-crescentus-CB15_mKate2_v01.h5",
        )
        config.set("pos1", "ModelWeights_None", f"{path_model_weights}")

    config.validate_id_section("pos1", basic=False)

    # We save to file
    config.to_file(tmp_dir)
    with pytest.raises(FileExistsError):
        config.to_file(tmp_dir, overwrite=False)

    # new config from file
    with pytest.raises(FileNotFoundError):
        new_config = Config.from_file(fname=os.path.join(tmp_dir, "settings.ono"))
    new_config = Config.from_file(fname=os.path.join(tmp_dir, "settings.ini"))

    # check for equality and name
    assert new_config.fname == "settings.ini"
    assert new_config == config

    # Mother Machine Settings
    # -----------------------

    # get the class
    config = Config(fname="settings.ini", general={"DataType": "Mother_Machine"})

    # check if we have the correct machine type
    assert config.get("General", "DataType") == "Mother_Machine"

    # set and id section and check that we don't have the remove border option
    config.set_id_section("pos1")
    config.set("pos1", "SegmentationClass", "StarDistSegmentation")

    # check all the id section tests
    config.validate_id_section("pos1", basic=True)

    assert config.get("pos1", "RemoveBorder", fallback=None) is None
