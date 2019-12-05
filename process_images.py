from imaging_system_responsefunction import get_response_function_and_log_irradiance, w
import cv2
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_SIZE = 5000


def sample_images(images, SAMPLE_SIZE=SAMPLE_SIZE):
    random_choices = np.random.choice(np.prod(images[0].shape),
                                      size=SAMPLE_SIZE,
                                      replace=False)

    x, y = np.unravel_index(random_choices, images[0].shape)

    flattened_images = np.zeros(shape=(images.shape[0], SAMPLE_SIZE),
                                dtype=np.int)

    for row_ind, image in enumerate(images):
        for col_ind, (_x, _y) in enumerate(zip(x, y)):
            flattened_images[row_ind, col_ind] = image[_x, _y]

    return flattened_images


def hdr_image_generator(images,
                        exposures,
                        channels=1,
                        lamb=42000.,
                        vectorized_w=w):
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2],
                            channels)
    output_image = np.zeros(images[0].shape, dtype=np.float32)

    plt.figure(figsize=(6, 6))
    colors = ['b', 'g', 'r'] if channels == 3 else ['y']

    for channel in range(channels):
        print('Channel : %s' % colors[channel])
        layers = np.array([img[:, :, channel] for img in images])
        layers_sampled = sample_images(layers)
        g, lnE = get_response_function_and_log_irradiance(
            layers_sampled, exposures, lamb)

        plt.plot(g, label=colors[channel], color=colors[channel])

        radiance_map = np.sum(
            w(img[:, :, channel]) *
            (g[img.astype(np.int)[:, :, channel]] -
             np.log(exposures[ind]) / np.sum(w(img[:, :, channel])))
            for ind, img in enumerate(images))

        output_image[..., channel] = cv2.normalize(radiance_map,
                                                   None,
                                                   alpha=0,
                                                   beta=255,
                                                   norm_type=cv2.NORM_MINMAX)

    if channels > 2:
        template = images[len(images) // 2]
        for ch in range(channels):
            image_avg, template_avg = np.average(
                output_image[..., ch]), np.average(template[..., ch])
            print(colors[ch], image_avg, template_avg)
            image_sd, template_sd = np.std(output_image[..., ch]), np.std(
                template[..., ch])

            output_image[..., ch] -= image_avg
            output_image[..., ch] /= image_sd
            output_image[..., ch] *= template_sd
            output_image[..., ch] += template_avg

    output_image = cv2.normalize(output_image,
                                 None,
                                 alpha=0,
                                 beta=255,
                                 norm_type=cv2.NORM_MINMAX)

    plt.legend()
    plt.savefig('output/g.png')
    plt.close()

    return output_image.astype(int)