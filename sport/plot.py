import os

from matplotlib import pyplot as plt

from matplotlib.font_manager import fontManager

CHINESE_FONT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'GB2312.ttf'
)

fontManager.addfont(CHINESE_FONT_PATH)
plt.rcParams['font.sans-serif'] = 'KaiTi_GB2312'

def draw(data: list, savePath: str | None = None, ax = None):
    if ax is None:
        plt.figure()
        ax = plt
    xs = list(range(len(data)))
    ax.plot(xs, data)
    if savePath is not None:
        os.makedirs(os.path.dirname(savePath), exist_ok = True)
        ax.savefig(savePath, dpi = 500)
