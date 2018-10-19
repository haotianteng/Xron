#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 03:11:00 2018

@author: heavens
"""
import tensorflow as tf
from chiron import chiron_model as model
import os
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph

LOG_PATH = "/home/heavens/UQ/Chiron_project/Chiron/chiron/model/DNA_default/"
OUT_PATH = "/home/heavens/UQ/Chiron_project/Chiron/chiron/model/DNA_default/freeze.bp"
BATCH_SIZE = 400
SEGMENT_LEN = 400
OUT_OP_NAME = "fea_rs"
x = tf.placeholder(tf.float32, shape=[BATCH_SIZE,SEGMENT_LEN])
seq_length = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
training = tf.placeholder(tf.bool)
config_path = os.path.join(LOG_PATH,'model.json')
model_configure = model.read_config(config_path)

logits, ratio = model.inference(
                                x, 
                                seq_length, 
                                training=training,
                                full_sequence_len = SEGMENT_LEN,
                                configure = model_configure)
sess = tf.Session()
saver = tf.train.Saver()
checkpoint_path = tf.train.latest_checkpoint(LOG_PATH)
saver.restore(sess, checkpoint_path)
sub_graph = tf.graph_util.extract_sub_graph(sess.graph.as_graph_def(),["fea_rs"])
graph_io.write_graph(sub_graph,'.','tmp.pb')
freeze_graph.freeze_graph('./tmp.pb', '',
                          	False, checkpoint_path, OUT_OP_NAME,
                          	"save/restore_all", "save/Const:0",
                          	OUT_PATH, False, "")