import numpy as np
from numba import njit, prange
from numba.core import types
from numba.typed import Dict


@njit
def calc_hash(arr, cell_size):
    p1 = 10000019
    p2 = 10000071
    p3 = 10000103
    r, g, b = arr
    hash_index = (
        int(r // cell_size) * p1 + int(g // cell_size) * p2 + int(b // cell_size) * p3
    )
    return int(hash_index)


@njit
def create_hashes(arrays, cell_size):
    arrays_f32 = arrays.astype(np.float32)
    hash_dict = Dict.empty(
        key_type=types.int64,
        value_type=types.float32[:, :],
    )

    for arr in arrays_f32:
        hash = calc_hash(arr, cell_size)
        if hash not in hash_dict:
            hash_dict[hash] = np.expand_dims(arr, 0)
        else:
            value = hash_dict[hash]
            hash_dict[hash] = np.concatenate((value, np.expand_dims(arr, 0)), 0)
    return hash_dict


@njit(parallel=True)
def query(hashes, point, radius, cell_size=5):
    neighbors = []
    radius_buckets = int(radius // cell_size)

    for dx in prange(-radius_buckets, radius_buckets + 1):
        for dy in range(-radius_buckets, radius_buckets + 1):
            for dz in range(-radius_buckets, radius_buckets + 1):
                neighbor_point = point.copy()
                neighbor_point[0] += dx * cell_size
                neighbor_point[1] += dy * cell_size
                neighbor_point[2] += dz * cell_size
                neighbor_hash = calc_hash(neighbor_point, cell_size)

                if neighbor_hash in hashes:
                    for neighbor in hashes[neighbor_hash]:
                        if np.linalg.norm(point - neighbor) < radius:
                            neighbors.append(neighbor)

    # Convert neighbors list to numpy array
    result = np.zeros((len(neighbors), len(point)), dtype=np.float32)
    for i in range(len(neighbors)):
        result[i] = neighbors[i]

    return result


@njit(parallel=True)
def mean_shift(data, bandwidth=5, max_iter=100, tolerance=1e-3):
    cell_size = bandwidth // 2
    data_f32 = data.astype(np.float32)
    hashes = create_hashes(data_f32, cell_size)
    shifted_points = data_f32
    for _ in range(max_iter):
        new_points = np.zeros_like(shifted_points)
        for idx in prange(len(shifted_points)):
            point = shifted_points[idx]
            neighbors = query(hashes, point, bandwidth, cell_size)

            if neighbors.size > 0:
                mean = np.zeros_like(neighbors[0], dtype=np.float32)
                for neighbour in neighbors:
                    mean += neighbour
                mean /= len(neighbour)

            else:
                mean = point

            new_points[idx] = mean

        shift = 0.0
        for i in range(len(new_points)):
            shift = max(shift, np.linalg.norm(new_points[i] - shifted_points[i]))

        if shift < tolerance:
            break

        shifted_points = new_points

    return shifted_points


def calculate_cluster_centers_skip(shifted_points, merge_threshold=1e-1, decimals=0):
    rounded_points = np.round(shifted_points, decimals=decimals)
    unique_points = np.unique(rounded_points, axis=0)

    @njit(parallel=True)
    def helper(unique_points):
        merged_centers = []
        for idx in prange(len(unique_points)):
            center = unique_points[idx]
            if len(merged_centers) == 0:
                merged_centers.append(center)
            else:
                too_close = False
                for merged_center in merged_centers:
                    if np.linalg.norm(center - merged_center) < merge_threshold:
                        too_close = True
                        break
                if not too_close:
                    merged_centers.append(center)
        return merged_centers

    return np.array(helper(unique_points))


def calculate_cluster_centers_average(shifted_points, merge_threshold=1e-1, decimals=2):
    rounded_points = np.round(shifted_points, decimals=decimals)
    unique_points = np.unique(rounded_points, axis=0)

    @njit(parallel=True)
    def helper(unique_points):
        merged_centers = []
        for idx in prange(len(unique_points)):
            center = unique_points[idx]
            if len(merged_centers) == 0:
                merged_centers.append(center)
            else:
                merged = False
                for i in range(len(merged_centers)):
                    merged_center = merged_centers[i]
                    if np.linalg.norm(center - merged_center) < merge_threshold:
                        merged_centers[i] = (merged_center + center) / 2
                        merged = True
                        break
                if not merged:
                    merged_centers.append(center)

        return merged_centers

    return np.array(helper(unique_points))


def classify_image(image, cluster_centers):

    height, width, c = image.shape

    return np.argmin(
        np.linalg.norm(
            image.reshape(-1, c)[:, None, :] - cluster_centers[None, :, :], axis=2
        ),
        axis=1,
    ).reshape(height, width)


if __name__ == "__main__":
    image = np.random.randint(0, 255, (30, 30, 3))
    data = image.reshape(-1, 3)
    result = mean_shift(data, bandwidth=31, tolerance=1, max_iter=25)
    centers = calculate_cluster_centers_average(result, merge_threshold=25, decimals=0)
    classes = classify_image(image, centers)
    print(classes)
