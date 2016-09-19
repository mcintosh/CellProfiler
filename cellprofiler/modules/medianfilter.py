"""

<strong>Median filter</strong>

Reduce noise in an image

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.exposure
import skimage.filters
import skimage.morphology


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

        x_data = skimage.img_as_uint(x_data)

        #TODO: use actual structuring_element when 3D is available for skimage.filters
        selem = self.__structuring_element()

        y_data = numpy.zeros_like(x_data)

        for plane, image in enumerate(x_data):
            y_data[plane] = skimage.filters.rank.median(image, selem)

        y_data = skimage.exposure.rescale_intensity(y_data * 1.0)

        y = cellprofiler.image.Image(
            image=y_data,
            parent_image=x
        )

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

    def display(self, workspace, figure):
        figure.set_grids((1, 2))

        figure.gridshow(0, 0, workspace.display_data.x_data)

        figure.gridshow(0, 1, workspace.display_data.y_data)

    def __structuring_element(self):
        if self.structuring_element.shape == "ball":
            return skimage.morphology.disk(self.structuring_element.size)

        if self.structuring_element.shape == "cube":
            return skimage.morphology.square(self.structuring_element.size)

        if self.structuring_element.shape == "octahedron":
            return skimage.morphology.diamond(self.structuring_element.size)

        return self.structuring_element.value
