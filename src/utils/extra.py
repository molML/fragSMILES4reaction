# Write float as percent, including zero digit until precision
def as_percent_str(float, digits=1):
    x = round(float, digits+2)
    return format(x*100, f'.{digits}f') + r'\%'

import pandas as pd
from ..configs.constants import NOTATIONS_COLOR, NOTATIONS_NAME, NOTATIONS

notation_table = pd.DataFrame(
    [NOTATIONS_COLOR, NOTATIONS_NAME],
    index=['color','name']
).T
notation_table.index.name = 'notation'
notation_table=notation_table.loc[NOTATIONS] # Ordering rows.

from rdkit.Chem import Draw
import io
from PIL import Image

class DrawerBase:
    def __init__(self):
        self.sizeImg=(-1,-1)
        
    def _getDrawer(self):
        drawer=Draw.MolDraw2DCairo(*self.sizeImg)
        options=drawer.drawOptions()
        options.setBackgroundColour((0,0,0,0))
        options.clearBackground=True
        return drawer
        
    def __call__(self, m):
        drawer=self._getDrawer()
        drawer.DrawMolecule(m)
        drawer.FinishDrawing()
        data=drawer.GetDrawingText()
        bio = io.BytesIO(data)
        img = Image.open(bio)
        return img

# NOTE yeah ... i don't like seaborn.
# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, write_diagonal=False,**textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    import matplotlib
    import numpy as np

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not write_diagonal and i==j:
                texts.append('')
                continue
            
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts