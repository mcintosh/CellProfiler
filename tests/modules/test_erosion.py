import cellprofiler.modules.erosion
import cellprofiler.setting
import numpy.testing
import skimage.morphology

instance = cellprofiler.modules.erosion.Erosion()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "erosion"

    if image.dimensions is 3 or image.multichannel:
        module.structuring_element = cellprofiler.setting.StructuringElement(shape="ball", size=1)
    else:
        module.structuring_element = cellprofiler.setting.StructuringElement(shape="disk", size=1)

    module.run(workspace)

    actual = image_set.get_image("erosion")

    desired = skimage.morphology.erosion(image.pixel_data, module.structuring_element.value)

    numpy.testing.assert_array_equal(actual.pixel_data, desired)
