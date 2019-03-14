import tensorflow as tf
import sys
import pdb
def printTensors(pb_file,num_lines):

    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    ops =  graph.get_operations() if num_lines < 0 else graph.get_operations()[:num_lines]
    for op in ops:
        print(op.name)
        print(op.values())

    #tss = graph.as_graph_def().node if num_lines < 0 else graph.as_graph_def().node[:num_lines]
    #for ts in tss:
    #   pdb.set_trace()
    #   print (ts.name)
  

#printTensors("/home/host_tf-pose//models/graph/mobilenet_thin/graph_opt.pb")
printTensors(sys.argv[1],int(sys.argv[2]))
