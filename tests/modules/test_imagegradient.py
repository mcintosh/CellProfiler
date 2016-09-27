import cellprofiler.modules.imagegradient
import numpy
import numpy.testing
import skimage.filters.rank
import skimage.morphology

instance = cellprofiler.modules.imagegradient.ImageGradient()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "ImageGradient"

    if image.dimensions is 3:
        module.structuring_element.shape = "ball"

    module.run(workspace)

    actual = image_set.get_image("ImageGradient")

    data = image.pixel_data

    disk = skimage.morphology.disk(1)

    if image.dimensions is 3 or image.multichannel:
        expected = numpy.zeros_like(data)

        for z, img in enumerate(data):
            expected[z] = skimage.filters.rank.gradient(img, disk)
    else:
        expected = skimage.filters.rank.gradient(data, disk)

    numpy.testing.assert_array_equal(expected, actual.pixel_data)
