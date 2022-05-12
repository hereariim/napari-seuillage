"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

from enum import Enum
from subprocess import call
import numpy
import napari
from napari.types import ImageData
from magicgui import magic_factory, magicgui
from skimage import data
import sys

import numpy

import numpy as np
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from skimage.io import imread
from enum import Enum
from napari.utils.notifications import show_info


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


def do_otsu(layer: ImageData) -> ImageData:
    
    import cv2
    img = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh1

def do_sobel(layer: ImageData) -> ImageData:
    import cv2
    img = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
    gX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    # compute the gradient magnitude and orientation
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    return magnitude

@magic_factory(call_button="Run",radio_option={"widget_type": "RadioButtons",
                        "orientation": "vertical",
                        "choices": [("Otsu",1), ("Sobel",2)]})
def do_model_segmentation(
    layer: ImageData, radio_option=1
    ) -> ImageData:
    if radio_option==1:
        show_info('Succes !')
        return do_otsu(layer)
    if radio_option==2:
        show_info('Succes !')
        return do_sobel(layer)
