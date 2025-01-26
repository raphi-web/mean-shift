import numpy as np
from numba import njit, prange


@njit(parallel=True)
def spatial_mean_shift(
    image, win_size=11, color_radius=10, max_iter=100, threshold=1e-3, weights=None
):
    h, w, c = image.shape
    image_f32 = image.astype(np.float32)
    new_image = np.zeros_like(image, dtype=np.float32)
    if win_size % 2 == 0:
        raise ValueError("Window size uneven.")

    win_size_half = win_size // 2

    if weights is not None:
        if weights.shape != (win_size, win_size):
            raise ValueError("Weights shape must be equal to win_size")

    for _ in range(max_iter):
        converged = True
        for i in prange(h):
            for j in range(w):
                pixel = image_f32[i, j, :]
                mean = np.zeros_like(pixel)
                cnt = 0
                for di in range(-win_size_half, win_size_half + 1):
                    row = i + di
                    if 0 <= row < h:
                        for dj in range(-win_size_half, win_size_half + 1):
                            col = j + dj
                            if 0 <= col < w:
                                neighbour = image_f32[row, col, :]
                                if np.linalg.norm(pixel - neighbour) < color_radius:
                                    if weights is not None:
                                        neighbour *= weights[
                                            di + win_size_half, dj + win_size_half
                                        ]
                                    mean += neighbour
                                    cnt += 1
                if cnt > 0:
                    mean /= cnt

                    if np.linalg.norm(mean - pixel) < threshold:
                        new_image[i, j, :] = pixel
                    else:
                        new_image[i, j, :] = mean
                        converged = False

                else:
                    # if no neighbours in the color_radius
                    new_image[i, j, :] = pixel

        image_f32 = new_image.copy()
        if converged:
            break

    return new_image.astype(image.dtype)


if __name__ == "__main__":
    x = np.random.randint(0, 255, (100, 100, 3))
    result = spatial_mean_shift(x, win_size=31, color_radius=31, threshold=1)
