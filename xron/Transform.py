#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 20:55:28 2018

@author: heavens
"""

from tensorflow.python.tools import freeze_graph
import chiron.model as model

def parse_model(input_path,configure_name = None):
    if configure_name is None:
        configure_name = 'model.json'
    config = model.read_config(input_path+configure_name)
    return config
def model2pb(configure, out_path)