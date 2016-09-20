"""

<strong>Median filter</strong>

Reduce salt-and-pepper noise in an image while preserving borders. Use a structuring element with a small size to
remove small elements of noise. A larger size will remove larger elements of noise at the risk of can blurring other
image features.

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.exposure
import skimage.filters


class MedianFilter(cellprofiler.module.Module):
    category = "Volumetric"
    module_name = "MedianFilter"
    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            u"Input"
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
            u"Output",
            u"OutputImage"
        )

        self.structuring_element = cellprofiler.setting.StructuringElement()

    def settings(self):
        return [
            self.x_name,
            self.y_name,
            self.structuring_element
        ]

    def visible_settings(self):
        return [
            self.x_name,
            self.y_name,
            self.structuring_element
        ]

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        # x_data = skimage.img_as_uint(x_data)

        if x.dimensions is 3 or x.multichannel:
            selem = self.__structuring_element()

            y_data = numpy.zeros_like(x_data)

            for plane, image in enumerate(x_data):
                y_data[plane] = skimage.filters.rank.median(image, selem)
        else:
            y_data = skimage.filters.rank.median(x_data, self.structuring_element.value)

        y_data = skimage.exposure.rescale_intensity(y_data * 1.0)

        y = cellprofiler.image.Image(
            image=y_data,
            parent_image=x,
            dimensions=x.dimensions
        )

        images.add(y_name, y)

        # import IPython
        # IPython.embed()

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = x.dimensions

            workspace.display_data.cmap = None if x.multichannel else "gray"

    def display(self, workspace, figure):
        figure.set_subplots((2, 1), dimensions=workspace.display_data.dimensions)

        figure.subplot_imshow(
            0,
            0,
            workspace.display_data.x_data,
            dimensions=workspace.display_data.dimensions,
            colormap=workspace.display_data.cmap
        )

        figure.subplot_imshow(
            1,
            0,
            workspace.display_data.y_data,
            dimensions=workspace.display_data.dimensions,
            colormap=workspace.display_data.cmap
        )

    def __structuring_element(self):
        if self.structuring_element.shape == "ball":
            self.structuring_element.shape = "disk"

        if self.structuring_element.shape == "cube":
            self.structuring_element.shape = "square"

        if self.structuring_element.shape == "octahedron":
            self.structuring_element.shape = "diamond"

        return self.structuring_element.value
