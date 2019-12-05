import numpy as np
import matplotlib.pyplot as plt


def _w(pixel_value):
    if pixel_value > 255 // 2:
        return 255 - pixel_value
    else:
        return pixel_value


w = np.vectorize(_w)


def get_response_function_and_log_irradiance(flattened_images,
                                             shutter_speeds,
                                             lamb,
                                             weight_function=w):
    """
        flattened_image : in range [0-255] grayscale images
        shutter_speeds : in seconds
        weight_function : a mapping between pixel values of the images to some weighted pixel values
        lamb: regularizer_factor 

        Basically an OLS fitting mathod ||Ax - b||^2 = O(A) where O is the objective function.
        The problem is sum_ij^{NP}(w(ij) * (g(ij) - lnE_i - lnt_j))^2 + lambda * reg, we need
        to construct matrix A and vector b in such a way that the norm we acquire results in
        the above sum.

        The implementation was presented in:
        
        Debevec, Paul E., and Jitendra Malik. 
        "Recovering high dynamic range radiance maps from photographs."
        ACM SIGGRAPH 2008 classes. ACM, 2008.

        Assumed:

        Z_min = 0
        Z_max = 255
    """

    max_pixel_difference = 255

    total_number_of_pixels = np.prod(flattened_images.shape)
    number_of_images = flattened_images.shape[0]
    number_of_pixels_per_image = flattened_images.shape[1]

    log_exposure_time = np.log(shutter_speeds)

    A = np.zeros(shape=(total_number_of_pixels + max_pixel_difference + 1,
                        max_pixel_difference + number_of_images))
    b = np.zeros(shape=(total_number_of_pixels + max_pixel_difference + 1, ))

    # Constructing the first part of the sum without regulaization

    counter = 0

    for row_ind in range(number_of_images):
        for col_ind in range(number_of_pixels_per_image):
            current_pixel_value = flattened_images[row_ind, col_ind]
            w = weight_function(current_pixel_value)
            A[counter, current_pixel_value] = w  # weight of g(ij)
            A[counter, max_pixel_difference + row_ind] = -w  # weight of lnE_i
            b[counter] = w * log_exposure_time[row_ind]  # weight of ln t_j
            counter += 1

    # Set the extra equation that g(Z_mid) = 0, Z_mid = (Z_min + Z_max) / 2
    # b = 0 at this counter

    A[counter, 127] = 1
    counter += 1

    # Regulaizer : lambda * g''(z) = lambda * (g(z+1) - 2g(z) + 2g(z-1))
    # the sum only goes from 1 - 254 -> due to the above equation

    for col_ind in range(1, max_pixel_difference):
        A[counter, col_ind] = lamb * weight_function(col_ind + 1)
        A[counter, col_ind + 1] = -2 * lamb * weight_function(col_ind)
        A[counter, col_ind + 2] = lamb * weight_function(col_ind - 1)
        counter += 1

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    plt.figure(figsize=(7, 7))
    plt.plot(np.matmul(A, x), b, 'bx', alpha=0.3)
    plt.xlabel('Solution')
    plt.ylabel('Actual values')
    plt.xlim(-600, 100)
    plt.ylim(-600, 100)
    plt.plot(b, b, 'r-', alpha=0.6)
    plt.savefig('output/plot.png')
    plt.close()

    return x[:max_pixel_difference + 1], x[max_pixel_difference:]
