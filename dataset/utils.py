import h5py
from numpy import shape, array

def store_single_hdf5(image, data_id, label, path):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    file = h5py.File(path + f"{data_id}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "image", shape(image), data=image
    )
    meta_set = file.create_dataset(
        "meta", shape(label), data=label
    )
    file.close()
    
def read_single_hdf5(image_id, path):
    """ Stores a single image to HDF5.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the HDF5 file
    file = h5py.File(path + f"{image_id}.h5", "r+")

    image = array(file["/image"])
    label = array(file["/meta"])

    return image, label

def store_many_hdf5(images, labels, data_id, path):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Create a new HDF5 file
    file = h5py.File(path + f"{num_images}_{data_id}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", shape(images), data=images
    )
    meta_set = file.create_dataset(
        "meta", shape(labels), data=labels
    )
    file.close()

def read_many_hdf5(num_images, data_id, path):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(path + f"{num_images}_{data_id}.h5", "r+")

    images = array(file["/images"])
    labels = array(file["/meta"])

    return images, labels
