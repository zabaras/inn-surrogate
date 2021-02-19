import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.ticker as ticker
import numpy as np
import torchvision.utils
plt.switch_backend('agg')
import scipy.io as io


def save_samples(save_dir, images, epoch, layer,plot, name, nrow=4, heatmap=True, cmap='jet'):
    """Save samples in grid as images or plots
    Args:
        images (Tensor): B x C x H x W
    """

    if images.shape[0] < 10:
        nrow = 2
        ncol = images.shape[0] // nrow
    else:
        ncol = nrow

    if heatmap:
        for c in range(images.shape[1]):
            # (11, 12)
            fig = plt.figure(1, (12, 12))
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(nrow, ncol),
                             axes_pad=0.3,
                             share_all=False,
                             cbar_location="right",
                             cbar_mode="single",
                             cbar_size="3%",
                             cbar_pad=0.1
                             )
            for j, ax in enumerate(grid):
                im = ax.imshow(images[j][c],cmap='jet', origin='lower',
                            interpolation='bilinear') 
                if j == 0:
                    ax.set_title('actual')
                elif j == 1:
                    ax.set_title('mean')
                else:
                    ax.set_title('sample %d'%(j-1))
                ax.set_axis_off()
                ax.set_aspect('equal')
            cbar = grid.cbar_axes[0].colorbar(im)
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.toggle_label(True)
            plt.subplots_adjust(top=0.95)
            plt.savefig(save_dir + '/{}_c{}_epoch{}_layer{}.pdf'.format(name, c, epoch,layer),
                        bbox_inches='tight')
            plt.close(fig)
    else:
        torchvision.utils.save_image(images, 
                          save_dir + '/fake_samples_epoch_{}.png'.format(epoch),
                          nrow=nrow,
                          normalize=True)

        
