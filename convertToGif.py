# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 02:46:19 2014

@author: ankur
"""

def sort_nicely( l ):
    import re
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l =sorted(l, key=alphanum_key )
    print l
    return l
  
def convert_images_to_gif():
    from images2gif import writeGif
    from PIL import Image
    import os

    list_images = (fn for fn in os.listdir('.') if fn.endswith('.png'))
    file_names = sort_nicely(list_images)
    images = [Image.open(fn) for fn in file_names]
    filename = "output.gif"
    writeGif(filename, images, duration=0.5)
    
convert_images_to_gif()
