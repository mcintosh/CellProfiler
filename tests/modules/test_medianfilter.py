import cellprofiler.modules.medianfilter
import cellprofiler.setting
import numpy
import numpy.testing
import skimage.exposure
import skimage.filters.rank
import skimage.morphology

instance = cellprofiler.modules.medianfilter.MedianFilter()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "opening"

    if image.dimensions is 3 or image.multichannel:
        module.structuring_element = cellprofiler.setting.StructuringElement("ball", 1)
    else:
        module.structuring_element = cellprofiler.setting.StructuringElement("disk", 1)

    selem = skimage.morphology.disk(1)

    module.run(workspace)

    actual = image_set.get_image("opening")

    if image.dimensions is 3 or image.multichannel:
        desired = numpy.zeros_like(image.pixel_data)

        for index, plane in enumerate(image.pixel_data):
            desired[index] = skimage.filters.rank.median(plane, selem)
    else:
        desired = skimage.filters.rank.median(image.pixel_data, selem)

    desired = skimage.exposure.rescale_intensity(desired * 1.0)

    numpy.testing.assert_array_equal(actual.pixel_data, desired)
