import matplotlib.pyplot as plt

from midap.utils import GUI_selector

from .semiautomated_cutout import SemiAutomatedCutout


def _chunked(lst, n):
    """Yield successive n-sized chunks from lst. Source - https://stackoverflow.com/a/312464"""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class SemiAutomatedCutoutSelect(SemiAutomatedCutout):
    """
    A class that performs the image cutout for the different channels in interactive mode with an additional step to select the chambers to include in the cutout
    """

    supported_setups = ["Mother_Machine"]

    def cut_corners(self, img, corners=None):
        """
        Given a single aligned image as array, it defines the corners that are used to cut out all images
        Additionally, it allows the user to select which chambers to include in the cutout
        :param img: Image to cut as array
        :param corners: Predefined corners for the cutout, if existing
        :returns: The corners of the cutout as tuple (left_x, right_x, lower_y, upper_y), where full range of the
                  image, i.e. the limits of the corners, are given by the total number of pixels.
        """
        previous_offsets = self.offsets.copy() if self.offsets is not None else []
        # interactive cutout of chambers
        corners = self.interactive_cutout(img, corners)
        self.corners_cut = tuple([int(i) for i in corners])

        # check that offsets are defined, for completeness sake
        if self.offsets is None:
            raise ValueError("Offsets must be defined for SemiAutomatedCutoutSelect!")

        # select the chambers to include in the cutout
        self.logger.info("Selecting Chambers to include")
        self.logger.debug(f"All offset idx: {list(range(len(self.offsets)))}")
        selected_offsets_idx = []
        chunk_size = 12
        n_chunks = len(self.offsets) // chunk_size + 1
        i_chunk = 0
        for batch in _chunked(list(enumerate(self.offsets)), chunk_size):
            figures = []
            indices = []
            marked_indices = []
            for i, offset in batch:
                base_corners = (
                    self.corners_cut[0] + offset,
                    self.corners_cut[1] + offset,
                    self.corners_cut[2],
                    self.corners_cut[3],
                )

                # perform the cutout of the first image
                chamber_img = self.do_cutout(img, base_corners)
                fig, ax = plt.subplots(figsize=(1, 2))
                ax.imshow(chamber_img)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(str(i))
                figures.append(fig)
                indices.append(i)
                if offset in previous_offsets:
                    marked_indices.append(i)

            if len(marked_indices) == 0:
                marked_indices = indices

            marked = GUI_selector(
                figures,
                labels=indices,
                title=f"Select chambers {i_chunk}/{n_chunks}",
                multiselect=True,
                marked=marked_indices,
            )
            selected_offsets_idx.extend(marked)
            i_chunk += 1
        self.logger.debug(f"Selected offset idx: {selected_offsets_idx}")
        self.offsets = [self.offsets[i] for i in selected_offsets_idx]
