import tensorflow as tf
import sys
import pdb
import os

height = sys.argv[2]
width = sys.argv[3]
def try1(graph_path, output_graph):
	with tf.gfile.GFile(graph_path, 'rb') as f:
	    graph_def = tf.GraphDef()
	    graph_def.ParseFromString(f.read())

	graph = tf.get_default_graph()
	#new_input = tf.placeholder(tf.float32, shape=(1 ,int(height),int(width), 3), name='const_input')
	new_input = tf.placeholder(tf.float32, shape=(1 ,int(width),int(height), 3), name='image')
	tf.import_graph_def(graph_def, name='', input_map={'image':new_input})
	sess = tf.Session(graph=graph)

	f = open("const_model_layers.txt","w")
	for node in graph.get_operations():
	     f.write(str(node.name))
	     f.write("\n")
	     f.write(str(node.values()))
	     f.write("\n")
	f.close()
	with tf.gfile.GFile(output_graph, "wb") as f:
	    f.write(graph.as_graph_def().SerializeToString())
	return 

if __name__ == '__main__':
	graph_path = sys.argv[1]
	try1(graph_path, os.path.splitext(graph_path)[0]+"_constant.pb")

