"""
A simple demo for std regularizaiton.
Reference:
Deep Convolutional Neural Networks with Spatial Regularization, Volume and Star-shape Prior for Image Segmentation.
https://arxiv.org/abs/2002.03989
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys


# gpu no.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import imageio


import logging

from std import STDLayer 

import torch
from torch.autograd import Variable


from utils import pascal_visualizer as vis
from utils import synthetic

try:
    import matplotlib.pyplot as plt
    figure = plt.figure()
    matplotlib = True
    plt.close(figure)
except:
    matplotlib = False
    pass



def plot_results(image, unary, output, label):

    logging.info("Show result")

    # Create visualizer
    myvis = vis.PascalVisualizer()

    # Transform id image to coloured labels
    coloured_label = myvis.id2color(id_image=label)

    unary_hard = np.argmax(unary, axis=2)
    coloured_unary = myvis.id2color(id_image=unary_hard)

    output = output[0]  # Remove Batch dimension
    output_hard = np.argmax(output, axis=0)
    coloured_std = myvis.id2color(id_image=output_hard)

    if matplotlib:
        # Plot results using matplotlib
        figure = plt.figure()
        figure.tight_layout()

        # Plot parameters
        num_rows = 2
        num_cols = 2

        ax = figure.add_subplot(num_rows, num_cols, 1)
        # img_name = os.path.basename(args.image)
        ax.set_title('Image ')
        ax.axis('off')
        ax.imshow(image)

        ax = figure.add_subplot(num_rows, num_cols, 2)
        ax.set_title('Label')
        ax.axis('off')
        ax.imshow(coloured_label.astype(np.uint8))

        ax = figure.add_subplot(num_rows, num_cols, 3)
        ax.set_title('Unary')
        ax.axis('off')
        ax.imshow(coloured_unary.astype(np.uint8))

        ax = figure.add_subplot(num_rows, num_cols, 4)
        ax.set_title('std Output')
        ax.axis('off')
        ax.imshow(coloured_std.astype(np.uint8))

        plt.show()
    else:
        out_img = np.concatenate(
            (image, coloured_label, coloured_unary, coloured_std),
            axis=1)

        imageio.imwrite("./out.png", out_img.astype(np.uint8))

        logging.info("Plot has been saved!")
    
    return



if __name__ == '__main__':

    # Load data
    image = imageio.imread('./data/2007_001288_0img.png')
    label = imageio.imread('./data/2007_001288_5labels.png')
    

    # Produce unary by adding noise to label
    unary = synthetic.augment_label(label, num_classes=21)
    num_classes = unary.shape[2]
    shape = image.shape[0:2]
    # make input pytorch compatible
    unary_var = unary.transpose(2, 0, 1)  # shape: [3, hight, width]
    # Add batch dimension to unary: [1, 21, height, width]
    unary_var = unary_var.reshape([1, num_classes, shape[0], shape[1]])

    # std layer
    std=STDLayer.STDLayer(21,10,4).cuda()  # (class number, iteration no., half size of kernel)
    logging.info("Start Computation.")
    #For gpu
    o=torch.Tensor(unary_var).cuda()

    with torch.no_grad():
        prediction=std(o) 

    output=prediction.data.cpu().numpy()

    # show result
    plot_results(image, unary, output, label)
    logging.info("std finished!")
