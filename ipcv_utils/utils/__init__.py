#-------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
#-------------------------------------------------------------------

def imshow(img, cmap="gray", vmin=0, vmax=1, frameon=False, zoom=1.0):

  dpi = float(matplotlib.rcParams['figure.dpi'])/zoom

  fig = plt.figure(figsize=[img.shape[1]/dpi, img.shape[0]/dpi],
                   frameon=frameon)
  ax = fig.add_axes([0, 0, 1, 1])
  ax.axis('off')
  ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

  # plt.savefig('contrast.png', dpi=300)
  plt.show()
#-------------------------------------------------------------------

def show(close=None, block=None):

  plt.show(close, block)
#-------------------------------------------------------------------

def imwrite(img, name, cmap="gray", vmin=0, vmax=1, frameon=False, zoom=1.0):

  # dpi = float(matplotlib.rcParams['figure.dpi'])/zoom

  # fig = plt.figure(figsize=[img.shape[1]/dpi, img.shape[0]/dpi],
  #                  frameon=frameon)
  fig = plt.figure()
  ax = fig.add_axes([0, 0, 1, 1])
  ax.axis('off')
  ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

  plt.savefig(name+'.png', dpi=300)
  # plt.show()