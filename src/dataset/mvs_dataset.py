from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode
import numpy as np
import re
import os


def read_pfm(filename):
    file = open(filename, 'rb')

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data


class MVSDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # Hypersim data parameter
            min_depth=1e-5,
            max_depth=65.0,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.rgb_i_d,
            **kwargs,
        )

    def _read_depth_file(self, rel_path):
        full_path = os.path.join(self.dataset_dir, rel_path)
        depth = np.load(full_path) if rel_path.endswith('.npy') else np.copy(read_pfm(full_path))
        depth = np.where(np.isnan(depth), np.zeros_like(depth), depth)
        return depth
