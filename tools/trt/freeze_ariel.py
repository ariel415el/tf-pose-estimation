import sys
from tensorflow.python.tools import freeze_graph
checkpoint_path = sys.argv[1]
freeze_graph.freeze_graph("", "", True, checkpoint_path, "Openpose/concat_stage7", "", "",checkpoint_path +  "_frozen.pb", False,"",  input_meta_graph=checkpoint_path + ".meta")


