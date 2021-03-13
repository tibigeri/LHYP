import numpy as np

def im_conv(dcm_image):
    im = dcm_image
    # filters pixels above 99%
    im = np.where(im<np.percentile(im,99), im, np.percentile(im,99))
    # filters pixels below 1%
    im = np.where(im<np.percentile(im,1),np.percentile(im,1),im)
    # rescales pixels between 0-255
    im = (im/np.amax(im))*255
    # converts pixel type to uint8
    im=im.astype(np.uint8)
    return im

    

