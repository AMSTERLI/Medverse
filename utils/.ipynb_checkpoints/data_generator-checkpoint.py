import os

import numpy as np
from scipy import ndimage
from skimage.transform import resize

# split_reim, convert_to_frequency_domain, convert_to_image_domain, join_reim

def split_reim(array):
    """Split a complex valued array into its real and imaginary parts along the last axis.

    Args:
      array(complex): An array of shape (...), where ... represents any number of dimensions.

    Returns:
      split_array(float): An array of shape (..., 2) containing the real and imaginary parts.
    """
    real = np.real(array)
    imag = np.imag(array)
    split_array = np.stack((real, imag), axis=-1)
    return split_array

def join_reim(array):
    """Join the real and imaginary parts from the last axis to form a complex array.

    Args:
      array(float): An array of shape (..., 2)

    Returns:
      joined_array(complex): A complex-valued array of shape (...)
    """
    joined_array = array[..., 0] + 1j * array[..., 1]
    return joined_array

def convert_to_frequency_domain(images):
    """Convert an array of images to their Fourier transforms.

    Args:
      images(float): An array of shape (batch_size, ..., 2), where ... represents spatial dimensions.

    Returns:
      spectra(float): An FFT-ed array of shape (batch_size, ..., 2)
    """
    n_dims = images.ndim - 2  # Subtract batch size and last dimension (real/imag)
    axes = tuple(range(1, 1 + n_dims))
    spectra = split_reim(np.fft.fftshift(np.fft.fftn(join_reim(images), axes=axes), axes=axes))
    return spectra

def convert_to_image_domain(spectra):
    """Convert an array of Fourier spectra to the corresponding images.

    Args:
      spectra(float): An array of shape (batch_size, ..., 2)

    Returns:
      images(float): An IFFT-ed array of shape (batch_size, ..., 2)
    """
    n_dims = spectra.ndim - 2  # Subtract batch size and last dimension (real/imag)
    axes = tuple(range(1, 1 + n_dims))
    images = split_reim(np.fft.ifftn(np.fft.ifftshift(join_reim(spectra), axes=axes), axes=axes))
    return images

def generate_undersampled_data(
        images,
        input_domain,
        output_domain,
        corruption_frac,
        enforce_dc,
        mask_axis = 1,
        random=True):
    """Generator that yields batches of undersampled input and correct output data for 3D images.

    For corrupted inputs, select each plane in k-space along the first axis with probability
    `corruption_frac` and set it to zero.

    Args:
      images(float): Numpy array of input images, of shape (num_images, n1, n2, n3)
      input_domain(str): The domain of the network input; 'FREQ' or 'IMAGE'
      output_domain(str): The domain of the network output; 'FREQ' or 'IMAGE'
      corruption_frac(float): Fraction of k-space planes to zero along the first axis
      enforce_dc(bool): Whether to enforce data consistency
      random(bool, optional): Whether to select random batches (Default value = True)

    Yields:
      inputs: Tuple of corrupted input data and ground truth output data, both numpy arrays
              of shape (b, n1, n2, n3, 2).
    """
    n_dims = images.ndim - 1  # Exclude batch size
    img_shape = images.shape[1:]  # Exclude batch size

    images = split_reim(images)
    spectra = convert_to_frequency_domain(images)
    
    # select the output
    batch_inds = range(images.shape[0])

    if input_domain in ('MAG', 'COMPLEX'):
        n_ch_in = 1
    else:
        n_ch_in = 2

    if output_domain in ('MAG', 'COMPLEX'):
        n_ch_out = 1
    else:
        n_ch_out = 2

    batch_img_shape = (images.shape[0],) + img_shape + (n_ch_in,)
    batch_out_shape = (images.shape[0],) + img_shape + (n_ch_out,)
    inputs = np.empty(batch_img_shape)
    outputs = np.empty(batch_out_shape)
    masks = np.empty(batch_img_shape)

    for idx_in_batch, j in enumerate(batch_inds):
        true_img = images[j:j+1, ...]  # Shape: (1, ..., 2)
        true_k = spectra[j:j+1, ...]   # Shape: (1, ..., 2)
        mask = np.ones_like(true_k)

        img_size = images.shape[1]
        num_points = int(img_size * corruption_frac)
        coord_list = np.random.choice(range(img_size), num_points, replace=False)

        corrupt_k = true_k.copy()
        for idx in coord_list:
            slices = [slice(None)] * corrupt_k.ndim
            slices[mask_axis] = idx  # Zero out along axis=1 at index idx. Could be 1, 2, 3
            corrupt_k[tuple(slices)] = 0
            mask[tuple(slices)] = 0

        corrupt_img = convert_to_image_domain(corrupt_k)

        nf = np.max(np.abs(corrupt_img))
        if nf == 0:
            nf = 1.0  # Avoid division by zero

        if input_domain == 'IMAGE':
            inputs[idx_in_batch] = corrupt_img / nf
        elif input_domain == 'FREQ':
            inputs[idx_in_batch] = corrupt_k / nf

        if output_domain == 'IMAGE':
            outputs[idx_in_batch] = true_img / nf
        elif output_domain == 'FREQ':
            outputs[idx_in_batch] = true_k / nf

        if enforce_dc:
            masks[idx_in_batch] = mask

    if enforce_dc:
        return (inputs, masks), outputs
    else:
        return inputs, outputs

import numpy as np
import scipy.ndimage as ndimage

def add_rotation_and_translations(sl, coord_list, angles, num_pix):
    """Add k-space rotations and translations to a 3D input volume.

    At each plane in `coord_list` in k-space, induce a rotation and translation.

    Args:
      sl (float): Numpy array of shape (n, n, n) containing input image data.
      coord_list (int): Numpy array of (num_points) k-space plane indices at which to induce motion.
      angles (float): Numpy array of rotation angles (in degrees) around x, y, z axes; shape (num_points, 3).
      num_pix (float): Numpy array of translations along x, y, z axes; shape (num_points, 3).

    Returns:
      sl_k_combined (float): Motion-corrupted k-space version of the input volume, of shape (n, n, n).
      sl_k_true (float): True k-space data after motion correction, of shape (n, n, n).
    """
    n = sl.shape[0]
    coord_list = np.concatenate([coord_list, [-1]])
    sl_k_true = np.fft.fftshift(np.fft.fftn(sl))

    sl_k_combined = np.zeros_like(sl_k_true, dtype='complex64')
    sl_k_combined[:coord_list[0], :, :] = sl_k_true[:coord_list[0], :, :]

    for i in range(len(coord_list) - 1):
        sl_rotated = sl.copy()

#         # Apply rotations around x, y, z axes
#         if angles[i, 0] != 0:
#             sl_rotated = ndimage.rotate(
#                 sl_rotated, angles[i, 0], axes=(1, 2), reshape=False, mode='nearest')
#         if angles[i, 1] != 0:
#             sl_rotated = ndimage.rotate(
#                 sl_rotated, angles[i, 1], axes=(0, 2), reshape=False, mode='nearest')
#         if angles[i, 2] != 0:
#             sl_rotated = ndimage.rotate(
#                 sl_rotated, angles[i, 2], axes=(0, 1), reshape=False, mode='nearest')

        # Apply translations along x, y, z axes
        sl_moved = ndimage.shift(sl_rotated, shift=num_pix[i], mode='nearest')

        sl_k_after = np.fft.fftshift(np.fft.fftn(sl_moved))

        if coord_list[i + 1] != -1:
            sl_k_combined[coord_list[i]:coord_list[i + 1], :, :] = sl_k_after[coord_list[i]:coord_list[i + 1], :, :]
            if coord_list[i] <= n // 2 < coord_list[i + 1]:
                sl_k_true = sl_k_after
        else:
            sl_k_combined[coord_list[i]:, :, :] = sl_k_after[coord_list[i]:, :, :]
            if coord_list[i] <= n // 2:
                sl_k_true = sl_k_after

    return sl_k_combined, sl_k_true

def generate_motion_data(
        images,
        input_domain,
        output_domain,
        mot_frac,
        max_trans,
        max_rot,
        batch_size=10):
    """Generator that yields batches of motion-corrupted input and correct output data for 3D images.

    For corrupted inputs, select some planes at which motion occurs; randomly generate and apply
    translations/rotations at those planes.

    Args:
      images (float): Numpy array of input images, of shape (num_images, n, n, n).
      input_domain (str): The domain of the network input; 'FREQ' or 'IMAGE'.
      output_domain (str): The domain of the network output; 'FREQ' or 'IMAGE'.
      mot_frac (float): Fraction of planes at which motion occurs.
      max_trans (float): Maximum fraction of image size for a translation (along each axis).
      max_rot (float): Maximum fraction of 360 degrees for a rotation (around each axis).
      batch_size (int, optional): Number of input-output pairs in each batch (Default value = 10).

    Yields:
      inputs: Numpy array of corrupted input data, shape (batch_size, n, n, n, 2).
      outputs: Numpy array of ground truth output data, shape (batch_size, n, n, n, 2).
    """
    img_shape = images.shape[1:]
    reim_images = images.copy()
    images = split_reim(images)
    spectra = convert_to_frequency_domain(images)


    n = images.shape[1]
    batch_inds = range(images.shape[0])

    inputs = []
    outputs = []

    for j in batch_inds:
        true_img = np.expand_dims(images[j, ...], axis=0)  # Shape: (1, n, n, n, 2)
        img_size = n
        num_points = int(mot_frac * n)
        coord_list = np.sort(np.random.choice(img_size, size=num_points, replace=False))

        num_pix = np.zeros((num_points, 3))
        angles = np.zeros((num_points, 3))

        max_trans_pix = n * max_trans
        max_rot_deg = 360 * max_rot

        # Random translations along x, y, z axes
        num_pix[:, 0] = np.random.uniform(-max_trans_pix, max_trans_pix, num_points)
        num_pix[:, 1] = np.random.uniform(-max_trans_pix, max_trans_pix, num_points)
        num_pix[:, 2] = np.random.uniform(-max_trans_pix, max_trans_pix, num_points)

        # Random rotations around x, y, z axes
        angles[:, 0] = np.random.uniform(-max_rot_deg, max_rot_deg, num_points)
        angles[:, 1] = np.random.uniform(-max_rot_deg, max_rot_deg, num_points)
        angles[:, 2] = np.random.uniform(-max_rot_deg, max_rot_deg, num_points)

        # Apply rotations and translations to the image
        corrupt_k, true_k = add_rotation_and_translations(
            reim_images[j, :, :, :], coord_list, angles, num_pix)
        corrupt_k = split_reim(np.expand_dims(corrupt_k, axis=0))
        true_k = split_reim(np.expand_dims(true_k, axis=0))

        corrupt_img = convert_to_image_domain(corrupt_k)

        nf = np.max(np.abs(corrupt_img))
        nf = nf if nf != 0 else 1.0  # Avoid division by zero

        if input_domain == 'FREQ':
            inputs.append(corrupt_k / nf)
        elif input_domain == 'IMAGE':
            inputs.append(corrupt_img / nf)

        if output_domain == 'FREQ':
            outputs.append(true_k / nf)
        elif output_domain == 'IMAGE':
            outputs.append(true_img / nf)

    inputs = np.concatenate(inputs, axis=0)
    outputs = np.concatenate(outputs, axis=0)

    return inputs, outputs

def interface_generate_undersampled_data(image, corruption_frac=0.15, mask_axis = 1):
    '''
    image (numpy array): shape (x,y,z), range [0,1]
    corruption_frac (float): range from [0.1, 0.2]
    mask_axis: from [1,2,3]
    
    return:
        shape (x,y,z)
    '''
    image[None,:]
    (inputs, masks), outputs = generate_undersampled_data(
    images=image[None,:],
    input_domain='IMAGE',
    output_domain='IMAGE',
    corruption_frac=corruption_frac,
    enforce_dc=True,
    mask_axis = mask_axis,
    random = False
    )
    return_image = inputs[0,:,:,:,0]
    
    # 0-1 normalize
    return_image = (return_image-return_image.min())/(return_image.max()-return_image.min())
    return return_image

import numpy as np
import scipy.ndimage as ndimage

def add_rotation_and_translations(sl, coord_list, angles, num_pix):
    """Add k-space rotations and translations to a 3D input volume.

    At each plane in `coord_list` in k-space, induce a rotation and translation.

    Args:
      sl (float): Numpy array of shape (n, n, n) containing input image data.
      coord_list (int): Numpy array of (num_points) k-space plane indices at which to induce motion.
      angles (float): Numpy array of rotation angles (in degrees) around x, y, z axes; shape (num_points, 3).
      num_pix (float): Numpy array of translations along x, y, z axes; shape (num_points, 3).

    Returns:
      sl_k_combined (float): Motion-corrupted k-space version of the input volume, of shape (n, n, n).
      sl_k_true (float): True k-space data after motion correction, of shape (n, n, n).
    """
    n = sl.shape[0]
    coord_list = np.concatenate([coord_list, [-1]])
    sl_k_true = np.fft.fftshift(np.fft.fftn(sl))

    sl_k_combined = np.zeros_like(sl_k_true, dtype='complex64')
    sl_k_combined[:coord_list[0], :, :] = sl_k_true[:coord_list[0], :, :]

    for i in range(len(coord_list) - 1):
        sl_rotated = sl.copy()

#         # Apply rotations around x, y, z axes
#         if angles[i, 0] != 0:
#             sl_rotated = ndimage.rotate(
#                 sl_rotated, angles[i, 0], axes=(1, 2), reshape=False, mode='nearest')
#         if angles[i, 1] != 0:
#             sl_rotated = ndimage.rotate(
#                 sl_rotated, angles[i, 1], axes=(0, 2), reshape=False, mode='nearest')
#         if angles[i, 2] != 0:
#             sl_rotated = ndimage.rotate(
#                 sl_rotated, angles[i, 2], axes=(0, 1), reshape=False, mode='nearest')

        # Apply translations along x, y, z axes
        sl_moved = ndimage.shift(sl_rotated, shift=num_pix[i], mode='nearest')

        sl_k_after = np.fft.fftshift(np.fft.fftn(sl_moved))

        if coord_list[i + 1] != -1:
            sl_k_combined[coord_list[i]:coord_list[i + 1], :, :] = sl_k_after[coord_list[i]:coord_list[i + 1], :, :]
            if coord_list[i] <= n // 2 < coord_list[i + 1]:
                sl_k_true = sl_k_after
        else:
            sl_k_combined[coord_list[i]:, :, :] = sl_k_after[coord_list[i]:, :, :]
            if coord_list[i] <= n // 2:
                sl_k_true = sl_k_after

    return sl_k_combined, sl_k_true

def generate_motion_data(
        images,
        input_domain,
        output_domain,
        mot_frac,
        max_trans,
        max_rot,
        batch_size=10):
    """Generator that yields batches of motion-corrupted input and correct output data for 3D images.

    For corrupted inputs, select some planes at which motion occurs; randomly generate and apply
    translations/rotations at those planes.

    Args:
      images (float): Numpy array of input images, of shape (num_images, n, n, n).
      input_domain (str): The domain of the network input; 'FREQ' or 'IMAGE'.
      output_domain (str): The domain of the network output; 'FREQ' or 'IMAGE'.
      mot_frac (float): Fraction of planes at which motion occurs.
      max_trans (float): Maximum fraction of image size for a translation (along each axis).
      max_rot (float): Maximum fraction of 360 degrees for a rotation (around each axis).
      batch_size (int, optional): Number of input-output pairs in each batch (Default value = 10).

    Yields:
      inputs: Numpy array of corrupted input data, shape (batch_size, n, n, n, 2).
      outputs: Numpy array of ground truth output data, shape (batch_size, n, n, n, 2).
    """
    img_shape = images.shape[1:]
    reim_images = images.copy()
    images = split_reim(images)
    spectra = convert_to_frequency_domain(images)


    n = images.shape[1]
    batch_inds = range(images.shape[0])

    inputs = []
    outputs = []

    for j in batch_inds:
        true_img = np.expand_dims(images[j, ...], axis=0)  # Shape: (1, n, n, n, 2)
        img_size = n
        num_points = int(mot_frac * n)
        coord_list = np.sort(np.random.choice(img_size, size=num_points, replace=False))

        num_pix = np.zeros((num_points, 3))
        angles = np.zeros((num_points, 3))

        max_trans_pix = n * max_trans
        max_rot_deg = 360 * max_rot

        # Random translations along x, y, z axes
        num_pix[:, 0] = np.random.uniform(-max_trans_pix, max_trans_pix, num_points)
        num_pix[:, 1] = np.random.uniform(-max_trans_pix, max_trans_pix, num_points)
        num_pix[:, 2] = np.random.uniform(-max_trans_pix, max_trans_pix, num_points)

        # Random rotations around x, y, z axes
        angles[:, 0] = np.random.uniform(-max_rot_deg, max_rot_deg, num_points)
        angles[:, 1] = np.random.uniform(-max_rot_deg, max_rot_deg, num_points)
        angles[:, 2] = np.random.uniform(-max_rot_deg, max_rot_deg, num_points)

        # Apply rotations and translations to the image
        corrupt_k, true_k = add_rotation_and_translations(
            reim_images[j, :, :, :], coord_list, angles, num_pix)
        corrupt_k = split_reim(np.expand_dims(corrupt_k, axis=0))
        true_k = split_reim(np.expand_dims(true_k, axis=0))

        corrupt_img = convert_to_image_domain(corrupt_k)

        nf = np.max(np.abs(corrupt_img))
        nf = nf if nf != 0 else 1.0  # Avoid division by zero

        if input_domain == 'FREQ':
            inputs.append(corrupt_k / nf)
        elif input_domain == 'IMAGE':
            inputs.append(corrupt_img / nf)

        if output_domain == 'FREQ':
            outputs.append(true_k / nf)
        elif output_domain == 'IMAGE':
            outputs.append(true_img / nf)

    inputs = np.concatenate(inputs, axis=0)
    outputs = np.concatenate(outputs, axis=0)

    return inputs, outputs


if __name__ == '__main__':
    # 创建伪造图像数据
    num_images = 128
    img_size = 128
    images = np.random.rand(num_images, img_size, img_size).astype(np.float32)
    # images = batch['image'][:,0,:].cpu().numpy()
    # 定义测试参数
    input_domain = 'IMAGE'
    output_domain = 'IMAGE'
    corruption_frac = 0.15 #0.1-0.2
    enforce_dc = True
    batch_size = 1


    # 初始化生成器
    inputs, outputs = generate_motion_data(
        images=images,
        input_domain=input_domain,
        output_domain=output_domain,
        mot_frac=0.025,
        max_trans=0.03,
        max_rot=0.00,
        batch_size=batch_size,
    )

    # 初始化生成器
    (inputs, masks), outputs = generate_undersampled_data(
        images=images,
        input_domain=input_domain,
        output_domain=output_domain,
        corruption_frac=corruption_frac,
        enforce_dc=enforce_dc,
        batch_size=batch_size,
        mask_axis = 1,
        random = False
    )

    print('inputs:',inputs.shape)
    print('masks:',masks.shape)
    print('outputs:',outputs.shape)

    # 可视化测试结果
    print("Inputs shape:", inputs.shape)
    print("Masks shape:", masks.shape)
    print("Outputs shape:", outputs.shape)

    # 展示第一个样本
    sub = 0
    slice_ = 40
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input (Corrupted)")
    plt.imshow(np.abs((inputs[sub, slice_, ..., 0])), cmap='gray')
    try:
        plt.subplot(1, 3, 2)
        plt.imshow(masks[sub, slice_, ..., 0], cmap='gray')
        plt.title("Mask")
    except:
        print('Mask:',masks)
    plt.subplot(1, 3, 3)
    plt.title("Output (Ground Truth)")
    plt.imshow(np.abs((outputs[sub, slice_, ..., 0])), cmap='gray')
    plt.show()

    # 可视化测试结果
    print("Inputs shape:", inputs.shape)
    print("Masks shape:", masks.shape)
    print("Outputs shape:", outputs.shape)

    # 展示第一个样本
    sub = 0
    slice_ = 40
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input (Corrupted)")
    plt.imshow(np.abs((inputs[sub, :,slice_, :, 0])), cmap='gray')
    try:
        plt.subplot(1, 3, 2)
        plt.imshow(masks[sub, :,slice_, :, 0], cmap='gray')
        plt.title("Mask")
    except:
        print('Mask:',masks)
    plt.subplot(1, 3, 3)
    plt.title("Output (Ground Truth)")
    plt.imshow(np.abs((outputs[sub, :,slice_, :, 0])), cmap='gray')
    plt.show()