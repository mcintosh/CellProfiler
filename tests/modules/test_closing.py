import cellprofiler.modules.closing
import cellprofiler.setting
import numpy.testing
import skimage.morphology

instance = cellprofiler.modules.closing.Closing()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "closing"

    if image.dimensions is 3 or image.multichannel:
        module.structuring_element = cellprofiler.setting.StructuringElement(shape="ball", size=1)
    else:
        module.structuring_element = cellprofiler.setting.StructuringElement(shape="disk", size=1)

    module.run(workspace)

    actual = image_set.get_image("closing")

    desired = skimage.morphology.closing(image.pixel_data, module.structuring_element.value)

    numpy.testing.assert_array_equal(actual.pixel_data, desired)
