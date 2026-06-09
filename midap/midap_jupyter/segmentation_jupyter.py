import os
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
import mpl_interactions.ipyplot as iplt
import numpy as np
import pandas as pd
import glob

from midap.utils import get_inheritors
from midap.segmentation import *
from midap.segmentation import base_segmentator
from midap.apps import segment_cells

import ipywidgets as widgets
from ipywidgets import interactive, Text, Password, Button, Output
from matplotlib.widgets import RadioButtons
from PIL import Image
from ipyfilechooser import FileChooser
import IPython as ip
import subprocess

from typing import Union, List


class SegmentationJupyter(object):
    """
    A class that performs image processing and segmentation based on midap
    """

    def __init__(self, path: Union[str, os.PathLike]):
        """
        Initializes the SegmentationJupyter
        :path: path to folder containing images
        """
        self.path = path
        self.path_midap = str(Path(__file__).parent.resolve().parent.parent)

        # existing folders
        self.path_data_input = self.path + "/input_data/"
        self.path_data = self.path + "/raw_im/"

        # folders created by class
        self.path_cut_base = Path(self.path).joinpath("cut_im/")
        self.path_seg_base = Path(self.path).joinpath("seg_im/")
        os.makedirs(self.path_cut_base, exist_ok=True)
        os.makedirs(self.path_seg_base, exist_ok=True)

    def get_input_dir(self):
        """
        Extracts input directory.
        """
        self.fc_file = FileChooser(self.path)
        self.fc_file.show_only_dirs = True
        self.fc_file.layout ={"width": "600px"}

    def get_input_files(self, path):
        """
        Extracts input file names.
        """
        self.file_selection = widgets.SelectMultiple(
            options=os.listdir(path),
            description="Files",
            disabled=False,
            layout={"height": "250px", "width": "600px"},
        )

        self.button = Button(description="Select")
        self.output = Output()

        def on_button_clicked(b):
            with self.output:
                self.chosen_files = self.file_selection.label
                self.chosen_dir = self.fc_file.selected
                #self.load_input_image()

        self.button.on_click(on_button_clicked)
        ip.display.display(self.file_selection)
        ip.display.display(self.button)

    def load_input_image(self, image_stack=False):
        """
        Loads selected image and extracts image dimensions.
        """

        # read image(s)
        self.imgs = []
        for f in self.chosen_files:
            path_chosen_img = Path(self.chosen_dir).joinpath(f)
            self.imgs.append(io.imread(path_chosen_img))

        # check if image dimensions align
        if len(list(set([i.shape for i in self.imgs]))) > 1:
            ip.display.display(
                ip.display.Markdown(
                    "**The image shapes do not match. Please select only images with the same image dimensions.**"
                )
            )
        elif len(list(set([i.shape for i in self.imgs]))) == 1:
            self.imgs = np.stack(self.imgs, axis=0)
            self.get_img_dims()
            self.get_img_dims_ix()

            # get indices of additional dimensions
            self.get_ix_add_dims()
            if image_stack==False:
                self.make_dropdowns_img_dims()
                ip.display.display(self.hbox_dropdowns)

    def get_img_dims(self):
        """
        Extracts height and width of an image.
        """
        img = Image.fromarray(self.imgs[0])
        self.img_height = img.height
        self.img_width = img.width

    def get_img_dims_ix(self):
        """
        Gets indices of img width and height in img shape.
        """
        self.img_shape = np.array(np.array(self.imgs).shape)

        if self.img_height != self.img_width:
            self.ix_height = np.where(self.img_shape == self.img_height)[0][0]
            self.ix_width = np.where(self.img_shape == self.img_width)[0][0]
        elif self.img_height == self.img_width:
            self.ix_height = np.where(self.img_shape == self.img_height)[0][0]
            self.ix_width = np.where(self.img_shape == self.img_width)[0][1]

    def get_ix_add_dims(self):
        """
        Gets axis of additional dimensions (number of frames and number of channels).
        """
        ix_dims = np.arange(len(np.array(self.imgs).shape))
        self.ix_diff = list(
            set(ix_dims).difference(set([self.ix_height, self.ix_width]))
        )

    def make_dropdowns_img_dims(self):
        """
        Makes dropdowns for number of channels and number of frames.
        """
        self.name_add_dims = ["num_channels", "num_images"]
        list_dropdowns = [
            self.make_dropdown(self.img_shape[ix_d], self.name_add_dims)
            for ix_d in self.ix_diff
        ]
        self.hbox_dropdowns = widgets.HBox(list_dropdowns)

    def make_dropdown(self, size_dim: int, name_add_dims: str):
        """
        Makes one dropdown per additional axis.
        :param size_dim: Length of additional axis.
        :param name_add_dims: Name of additional axis.
        """
        drop_options = name_add_dims
        dropdown = widgets.Dropdown(
            options=drop_options,
            layout=widgets.Layout(width="50%"),
            description="Dimension with length " + str(size_dim) + " is:",
            style={"description_width": "250px"},
        )
        return dropdown

    def spec_img_dims(self):
        """
        Specify image dimensions acc. to following pattern: (num_imgs, height, width, num_channels).
        """

        # get indices of frame and channel indentifier
        name_dims = [c.value for c in self.hbox_dropdowns.children]

        # check which axes are currently present and create dict for assignment of new axes
        self.dims_assign_dict = dict()
        self.axis_length_dict = dict()

        for md, ixd in zip(name_dims, self.ix_diff):
            self.dims_assign_dict[md] = ixd
            self.axis_length_dict[md] = self.img_shape[ixd]

    def align_img_dims(self):
        """
        Aligns image dimensions to following pattern: (num_imgs, height, width, num_channels).
        """

        # move axes to (num_frames, height, width, num_channels)
        self.imgs_clean = self.imgs.copy()
        self.imgs_clean = np.array(self.imgs_clean)

        if (
            "num_images" in self.dims_assign_dict.keys()
            and "num_channels" in self.dims_assign_dict.keys()
        ):
            new_ax_im_len = np.where(
                np.array(self.img_shape) == self.axis_length_dict["num_images"]
            )[0][0]
            self.imgs_clean = np.moveaxis(self.imgs_clean, new_ax_im_len, 0)

            new_ax_ch_len = np.where(
                np.array(self.imgs_clean.shape) == self.axis_length_dict["num_channels"]
            )[0][0]
            self.imgs_clean = np.moveaxis(self.imgs_clean, new_ax_ch_len, -1)

        if (
            "num_images" in self.dims_assign_dict.keys()
            and "num_channels" not in self.dims_assign_dict.keys()
        ):
            self.imgs_clean = np.moveaxis(
                self.imgs_clean, self.dims_assign_dict["num_images"], 0
            )

        if (
            "num_images" not in self.dims_assign_dict.keys()
            and "num_channels" in self.dims_assign_dict.keys()
        ):
            self.imgs_clean = np.moveaxis(
                self.imgs_clean, self.dims_assign_dict["num_channels"], -1
            )

        # add axis in case one is missing
        if "num_images" not in self.dims_assign_dict.keys():
            self.imgs_clean = np.expand_dims(self.imgs_clean, axis=0)

        if "num_channels" not in self.dims_assign_dict.keys():
            self.imgs_clean = np.expand_dims(self.imgs_clean, axis=-1)

    def show_example_image(self, img: np.ndarray):
        """
        Displays example image.
        """
        _, self.ax = plt.subplots()
        self.ax.imshow(img)
        plt.show()

    def select_channel(self):
        """
        Creates a figure linked to a dropdown to select channel.
        """

        def f(a, c):
            _, ax1 = plt.subplots()
            ax1.imshow(self.imgs_clean[int(c), :, :, int(a)])
            ax1.set_xticks([])
            ax1.set_yticks([])
            plt.title("Channel: " + str(a))
            plt.show()

        self.output_sel_ch = interactive(
            f,
            a=widgets.Dropdown(
                options=np.arange(self.imgs_clean.shape[-1]),
                layout=widgets.Layout(width="50%"),
                description="Channel",
            ),
            c=widgets.IntSlider(
                min=0,
                max=self.imgs_clean.shape[0] - 1,
                description="Image ID",
            ),
        )

    def set_channel(self):
        """
        Set selected channel based on label of dropdown.
        """
        self.selected_ch = int(self.output_sel_ch.children[0].label)
        self.imgs_sel_ch = self.imgs_clean[:, :, :, self.selected_ch]
        self.imgs_sel_ch = np.expand_dims(self.imgs_sel_ch, -1)

    def get_corners_cutout(self):
        """
        Gets axis limits for current zoom-in.
        """
        xlim = [int(xl) for xl in self.ax.get_xlim()]
        ylim = [int(yl) for yl in self.ax.get_ylim()]

        self.x_min = np.min(xlim)
        self.x_max = np.max(xlim)

        self.y_min = np.min(ylim)
        self.y_max = np.max(ylim)

    def make_cutouts(self):
        """
        Generates cutouts for all images.
        """
        self.imgs_cut = np.array(
            [
                img[self.y_min : self.y_max, self.x_min : self.x_max, 0]
                for img in self.imgs_sel_ch
            ]
        )

    def show_all_cutouts(self):
        """
        Displays all cutouts with slider.
        """

        def f(i):
            _, ax1 = plt.subplots()
            ax1.imshow(self.imgs_cut[int(i)])
            ax1.set_xticks([])
            ax1.set_yticks([])
            plt.show()

        self.output_all_cuts = interactive(
            f,
            i=widgets.IntSlider(
                min=0,
                max=self.imgs_cut.shape[0] - 1,
                description="Image ID",
            ),
        )

    def save_cutouts(self):
        """
        Saves all cutouts in new folder.
        """

        # update path to cutout images and create dir if necessary
        self.path_cut = Path(self.path_cut_base).joinpath(
            Path(self.chosen_files[0]).stem
        )
        os.makedirs(self.path_cut, exist_ok=True)

        for f, cut in zip(self.chosen_files, self.imgs_cut):
            cut_scale = self.scale_pixel_val(cut)
            io.imsave(self.path_cut.joinpath(Path(f).stem + "_cut.png"), cut_scale, check_contrast=False)

    def get_segmentation_models(self):
        """
        Collects all json files and combines model information in table.
        """
        files = glob.glob(
            self.path_midap + "/model_weights/**/model_weights*.json", recursive=True
        )

        df = pd.read_json(files[0])

        for f in files[1:]:
            df_new = pd.read_json(f)
            df = pd.concat([df, df_new], axis=1)

        self.df_models = df.T


    def display_segmentation_models(self):
        """
        Displays availbale models in interactive table.
        """

        def f(a, b):
            self.df_models_filt = self.df_models[
                self.df_models["species"].isin(a) & self.df_models["marker"].isin(b)
            ]

            self.df_models_filt2 = self.df_models_filt.drop(columns=['nn_type_alias'])
            display(self.df_models_filt2)

        self.outp_interact_table = interactive(
            f,
            a=widgets.SelectMultiple(
                options=self.df_models["species"].unique(),
                value=list(self.df_models["species"].unique()),
                layout=widgets.Layout(width="50%"),
                description="Species",
            ),
            b=widgets.SelectMultiple(
                options=self.df_models["marker"].unique(),
                value=list(self.df_models["marker"].unique()),
                layout=widgets.Layout(width="50%"),
                description="Marker/Type",
            ),
        )

    def select_segmentation_models(self):
        """
        Selects segmentation models based on output of interactive table.
        """
        self.all_chosen_seg_models = {}
        for nnt in self.df_models_filt.nn_type_alias.unique():
            self.all_chosen_seg_models[nnt] = list(
                self.df_models_filt[self.df_models_filt.nn_type_alias == nnt].index
            )

    def run_all_chosen_models(self):
        """
        Runs all pretrained models of chosen model types.
        """
        self.dict_all_models = {}
        self.dict_all_models_label = {}
        for nnt, models in self.all_chosen_seg_models.items():
            self.select_segmentator(nnt)
            for model in models:
                model_name = "_".join((model).split("_")[2:])
                self.pred.segment_images_jupyter(
                    self.imgs_cut, model_name, clean_border=False
                )
                self.dict_all_models["{}_{}".format(nnt, model)] = self.pred.seg_bin
                self.dict_all_models_label[
                    "{}_{}".format(nnt, model)
                ] = self.pred.seg_label

    def select_segmentator(self, segmentation_class: str):
        """
        Selects segmentator based on segmentation class.
        :param segmentation_class: Name of segmentation class.
        """
        if segmentation_class == "OmniSegmentationJupyter":
            path_model_weights = Path(self.path_midap).joinpath(
                "model_weights", "model_weights_omni"
            )
        else:
            path_model_weights = Path(self.path_midap).joinpath(
                "model_weights", "model_weights_legacy"
            )

        # define variables
        postprocessing = False
        network_name = None
        img_threshold = 255


        # get the right subclass
        class_instance = None

        segmentation_subclasses = get_inheritors(base_segmentator.SegmentationPredictor)
        jupyter_seg_cls = [s for s in segmentation_subclasses if "Jupyter" in s.supported_setups]
        
        #for subclass in get_inheritors(base_segmentator.SegmentationPredictor):
        for subclass in jupyter_seg_cls:
            if subclass.__name__ == segmentation_class:
                class_instance = subclass

        # throw an error if we did not find anything
        if class_instance is None:
            raise ValueError(f"Chosen class does not exist: {segmentation_class}")

        # get the Predictor
        self.pred = class_instance(
            path_model_weights=path_model_weights,
            postprocessing=postprocessing,
            model_weights=network_name,
            img_threshold=img_threshold,
        )

    def compare_segmentations(self):
        """
        Displays two segmentations side-by-side for comparison of different pretrained models.
        """

        def f(a, b, c):
            fig = plt.figure(figsize=(12, 12))

            sem_seg_a = self.dict_all_models[a][int(c)]
            sem_seg_a = np.ma.masked_where(sem_seg_a == 0, sem_seg_a)

            sem_seg_b = self.dict_all_models[b][int(c)]
            sem_seg_b = np.ma.masked_where(sem_seg_b == 0, sem_seg_b)

            inst_seg_a = self.dict_all_models_label[a][int(c)]
            inst_seg_a = np.ma.masked_where(inst_seg_a == 0, inst_seg_a)

            inst_seg_b = self.dict_all_models_label[b][int(c)]
            inst_seg_b = np.ma.masked_where(inst_seg_b == 0, inst_seg_b)


            ax1 = fig.add_subplot(221)
            plt.imshow(sem_seg_a, cmap='tab20')
            ax1.set_xticks([])
            ax1.set_yticks([])
            plt.title('Model 1 (semantic segmentation)')

            ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
            plt.imshow(sem_seg_b, cmap='tab20')
            plt.title('Model 2 (semantic segmentation)')

            ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
            plt.imshow(inst_seg_a, cmap='tab20')
            plt.title('Model 1 (instance segmentation)')

            ax4 = fig.add_subplot(224, sharex=ax1, sharey=ax1)
            plt.imshow(inst_seg_b, cmap='tab20')
            plt.title('Model 2 (instance segmentation)')

            plt.show()

        self.output_seg_comp = interactive(
            f,
            a=widgets.Dropdown(
                options=self.dict_all_models.keys(),
                layout=widgets.Layout(width="50%"),
                description="Model 1",
            ),
            b=widgets.Dropdown(
                options=self.dict_all_models.keys(),
                layout=widgets.Layout(width="50%"),
                description="Model 2",
            ),
            c=widgets.IntSlider(
                min=0,
                max=(len(list(self.dict_all_models.values())[0]) - 1),
                description="Image ID",
            ),
        )

    def display_buttons_weights(self):
        """
        Displays all used models for segmentation to select best model.
        """
        self.out_weights = widgets.RadioButtons(
            options=list(self.dict_all_models.keys()),
            description="Model weights:",
            disabled=False,
            layout=widgets.Layout(width="100%"),
        )

    def load_add_files(self):
        """
        Loads additional (full) image stack.
        """

        def f(a):
            if a == True:
                self.fc_add_file = FileChooser(self.path)
                self.fc_add_file.show_only_dirs = True
                self.fc_add_file.layout ={"width": "600px"}
                ip.display.display(self.fc_add_file)

        self.out_add_file = interactive(
            f,
            a=widgets.Checkbox(
                value=False,
                description="Do you want to select an additional dataset for the segmentation?",
            ),
        )

    def segment_all_images(self, model_name: str):
        """
        Segments all images for given model type and selected model weights.
        :param model_name: Name of chosen trained model.
        """
        self.pred.segment_images_jupyter(self.imgs_cut, model_name, clean_border=False)

    def process_images(self):
        """
        Processes all images after loading the full image stack.
        """
        if self.out_add_file.children[0].value == True:
            self.chosen_files = os.listdir(self.fc_add_file.selected)
            self.chosen_dir = self.fc_add_file.selected
            self.load_input_image(image_stack=True)
            #self.get_img_dims_ix()
            #self.spec_img_dims()
            self.align_img_dims()
            self.set_channel()
            self.make_cutouts()
            self.save_cutouts()

        self.select_segmentator(self.out_weights.label.split("_")[0])
        self.segment_all_images(("_").join(self.out_weights.label.split("_")[3:]))
        self.save_segs()

    def save_segs(self):
        """
        Saves all segmentations in new folder.
        """

        self.path_seg = Path(self.path_seg_base).joinpath(
            Path(self.chosen_files[0]).stem
        )
        os.makedirs(self.path_seg, exist_ok=True)

        segs = np.array(self.pred.seg_label)

        for f, seg in zip(self.chosen_files, segs):
            io.imsave(self.path_seg.joinpath(Path(f).stem + "_seg.tif"), seg, check_contrast=False)

    def get_usern_pw(self):
        """
        Get username and password for upload to polybox.
        """

        self.out_usern = Text(
            value="", placeholder="", description="Username:", disabled=False
        )

        self.out_passw = Password(
            value="", placeholder="", description="Password:", disabled=False
        )

        self.button = Button(description="Confirm")
        self.output = Output()

        def on_button_clicked(b):
            with self.output:
                arg1 = self.out_usern.value
                arg2 = self.out_passw.value
                arg3 = str(self.path_seg).rstrip("/")
                subprocess.call(
                    "./upload_polybox.sh "
                    + str(arg1)
                    + " "
                    + str(arg2)
                    + " "
                    + str(arg3),
                    shell=True,
                )

        self.button.on_click(on_button_clicked)

    def scale_pixel_val(self, img):
        """
        Rescale the pixel values of the image
        :param img: The input image as array
        :returns: The images with pixels scales to standard RGB values
        """
        img_scaled = (255 * ((img - np.min(img)) / np.max(img - np.min(img)))).astype(
            "uint8"
        )
        return img_scaled
