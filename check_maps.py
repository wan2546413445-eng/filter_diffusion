import h5py
import numpy as np

map_file = "/mnt/SSD/wsy/projects/HFS-SDE-master/data/multicoil_train_knee/maps/file1000059.h5"
map_key = "s_maps"
slice_idx = 0

with h5py.File(map_file, "r") as f:
    print("keys:", list(f.keys()))
    data = f[map_key]
    print(f"{map_key} full shape:", data.shape)
    print(f"{map_key} dtype:", data.dtype)

    x = data[slice_idx]
    print(f"slice[{slice_idx}] shape:", x.shape)
    print(f"slice[{slice_idx}] dtype:", x.dtype)

    print("is complex:", np.iscomplexobj(x))
    if np.iscomplexobj(x):
        print("real min/max:", np.real(x).min(), np.real(x).max())
        print("imag min/max:", np.imag(x).min(), np.imag(x).max())
        print("abs  min/max:", np.abs(x).min(), np.abs(x).max())
    else:
        print("value min/max:", x.min(), x.max())

    print("ndim:", x.ndim)
    for i, s in enumerate(x.shape):
        print(f"dim {i}: {s}")