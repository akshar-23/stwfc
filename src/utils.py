import numpy as np

def islistlike(obj):
    return isinstance(obj, list) or isinstance(obj, np.ndarray)

def tuplify(listlike):
    if islistlike(listlike[0]):
        return tuple(tuplify(x) for x in listlike)
    return tuple(listlike)

def pad(grid, pad_values=("X", "X", "X"), pad_widths=(1, 1, 1)):
    expanded_pad_values = []
    for dim_vals in pad_values:
        if not islistlike(dim_vals):
            expanded_pad_values.append((dim_vals, dim_vals))
        else:
            expanded_pad_values.append(dim_vals)
    while(len(expanded_pad_values) < len(grid.shape)):
        expanded_pad_values.append(expanded_pad_values[0])
    expanded_pad_values = tuplify(expanded_pad_values)

    expanded_pad_widths = []
    for dim_widths in pad_widths:
        if not islistlike(dim_widths):
            expanded_pad_widths.append((dim_widths, dim_widths))
        else:
            expanded_pad_widths.append(dim_widths)
    expanded_pad_widths = tuplify(expanded_pad_widths)

    L = grid
    for i in range(len(pad_widths) - 1, -1, -1):
        set_pad_widths = np.zeros((len(L.shape), 2), dtype=int)
        set_pad_widths[i] = expanded_pad_widths[i]
        L = np.pad(
            L,
            pad_width=tuplify(set_pad_widths),
            constant_values=expanded_pad_values,
        )
    return L


def unpad_3D(grid, pad_widths=(1, 1, 1)):
    return grid[
        pad_widths[0] : -pad_widths[0],
        pad_widths[1] : -pad_widths[1],
        pad_widths[2] : -pad_widths[2],
    ]