import cellprofiler.modules.medialaxis
import numpy.testing
import skimage.morphology

instance = cellprofiler.modules.medialaxis.MedialAxis()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "MedialAxis"

    module.run(workspace)

    actual = image_set.get_image("MedialAxis")

    desired = skimage.morphology.medial_axis(image.pixel_data)

    numpy.testing.assert_array_equal(actual.pixel_data, desired)
