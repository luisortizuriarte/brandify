from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Preset location aliases and anchor alignments
_LOC_ANCHORS = {
    'top right': 'NE',
    'upper right': 'NE',
    'top left': 'NW',
    'upper left': 'NW',
    'bottom right': 'SE',
    'lower right': 'SE',
    'bottom left': 'SW',
    'lower left': 'SW',
    'center': 'C',
}


def _load_image(img_input):
    """Load image from path, file-like object, PIL Image, or numpy array."""
    if isinstance(img_input, np.ndarray):
        return img_input
    # Support PIL.Image without requiring Pillow as a hard import
    if hasattr(img_input, 'size') and hasattr(img_input, 'mode') and not isinstance(img_input, (str, bytes, Path)):
        return np.asarray(img_input)
    return plt.imread(img_input)


def _compute_bbox(fig, img, loc, size, pad):
    """Compute in-bounds bounding box preserving image aspect ratio."""
    fig_w, fig_h = fig.get_size_inches()
    img_h, img_w = img.shape[:2]

    # Ratio of width to height for image and figure
    img_ar = img_w / img_h
    fig_ar = fig_w / fig_h

    # Compute width and height fractionally so physical aspect ratio is preserved
    if img_ar >= 1.0:
        # Wide or square logo: constrain width by size fraction
        w_frac = size
        h_frac = (size * fig_ar) / img_ar
    else:
        # Tall logo: constrain height by size fraction
        h_frac = size
        w_frac = (size * img_ar) / fig_ar

    # Ensure width and height do not exceed available figure canvas
    w_frac = min(w_frac, max(0.01, 1.0 - 2 * pad))
    h_frac = min(h_frac, max(0.01, 1.0 - 2 * pad))

    if loc in ('upper right', 'top right'):
        x = 1.0 - w_frac - pad
        y = 1.03 - h_frac
    elif loc in ('upper left', 'top left'):
        x = pad
        y = 1.03 - h_frac
    elif loc in ('lower right', 'bottom right'):
        x = 1.0 - w_frac - pad
        y = pad
    elif loc in ('lower left', 'bottom left'):
        x = pad
        y = pad
    elif loc == 'center':
        x = (1.0 - w_frac) / 2.0
        y = (1.0 - h_frac) / 2.0
    else:
        valid_locs = ['top right', 'top left', 'bottom right', 'bottom left', 'center']
        raise ValueError(f"Invalid loc '{loc}'. Expected one of: {valid_locs}")

    return [x, y, w_frac, h_frac]


def embed_logo(fig, img_file=None, loc='top right', bbox=None, size=0.35, pad=0.02, alpha=None, zorder=1000, img=None):
    """ Embed an image to an existing matplotlib figure object. 
        
        Input
        -----
        fig (obj): Matplotlib figure object
        img_file (str | Path | array | Image): File path, file-like object,
            PIL Image, or numpy array of the image to be embedded.
        loc (str): Location to embed figure. Can be any of the following:
        
                    top right    (or upper right) [Default]
                    top left     (or upper left)
                    bottom right (or lower right)
                    bottom left  (or lower left)
                    center

        Defaults to "top right".

        bbox (iter): If a bbox is specified, provide exact location and size of
        image axis to be drawn. when specified, loc keyword argument is
        ignored. Default None. The bbox iterable follows the following
        structure:
            [x_coordinate, y_coordinate, x_length, y_length]

        where all values are in fractional figure units. 

        size (float): Maximum dimension (fraction of figure width/height) 
        allocated for the logo when bbox is None. Defaults to 0.35.

        pad (float): Fractional padding from the figure border when bbox is None.
        Defaults to 0.02.

        alpha (float, optional): Transparency level between 0.0 (transparent)
        and 1.0 (opaque). Default None.

        zorder (int): Drawing order of the embedded logo axis. Defaults to 1000.

        img: Alias for img_file.
    """
    if not hasattr(fig, 'add_axes') or not hasattr(fig, 'get_size_inches'):
        raise TypeError("fig must be a matplotlib Figure object.")

    # Accept either img or img_file
    source_img = img if img is not None else img_file
    if source_img is None:
        raise ValueError("An image file path, file object, array, or PIL Image must be provided.")

    # Load image data
    loaded_img = _load_image(source_img)

    # Determine bounding box and anchor
    if bbox is not None:
        try:
            bbox_val = [float(v) for v in bbox]
            if len(bbox_val) != 4:
                raise ValueError
        except (TypeError, ValueError) as err:
            raise ValueError("bbox must be an iterable of 4 numbers [x, y, width, height]") from err
        anchor = 'C'
    else:
        norm_loc = loc.strip().lower() if isinstance(loc, str) else ''
        if norm_loc not in _LOC_ANCHORS:
            valid_keys = ['top right', 'top left', 'bottom right', 'bottom left', 'center']
            raise ValueError(f"Invalid loc '{loc}'. Expected one of {valid_keys} or specify 'bbox'.")
        bbox_val = _compute_bbox(fig, loaded_img, norm_loc, size=size, pad=pad)
        anchor = _LOC_ANCHORS[norm_loc]

    ax = fig.add_axes(bbox_val, 
                      anchor=anchor,
                      zorder=zorder)
    
    # Create image object into ax obj
    ax.imshow(loaded_img, aspect='equal', alpha=alpha)
    ax.axis('off')

    return fig
