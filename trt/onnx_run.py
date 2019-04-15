path = "/briefcam/Projects/ArielE/trained_models/multi-gpu/mobilenet_thin_batch_48_lr_0.0001_432x368_gpu1_from_reg_48_129k/model-150002/model-150002_frozen_opt_constant.onnx"
# path = "/briefcam/Projects/ArielE/trained_models/multi-gpu/mobilenet_thin_batch_48_lr_0.0001_432x368_gpu1_from_reg_48_129k/model-150002/pretrained_graph_opt_constant_432x368.onnx"
import onnxruntime as rt
import numpy
import cv2
import numpy as np
sess = rt.InferenceSession(path)
im = cv2.imread("/briefcam/Projects/p1.jpg").astype(np.uint8)
im = np.array([cv2.resize(im,(432,368))])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: im.astype(numpy.float32)})[0]
# import pdb;pdb.set_trace()
print(pred_onx.shape)
# cv2.imwrite("/briefcam/Projects/im_15.png",pred_onx[0,:,:,15]*255)
# cv2.imwrite("/briefcam/Projects/im_16.png",pred_onx[0,:,:,16]*255)
# cv2.imwrite("/briefcam/Projects/im_17.png",pred_onx[0,:,:,17]*255)
# cv2.imwrite("/briefcam/Projects/im_18.png",pred_onx[0,:,:,18]*255)
# cv2.imwrite("/briefcam/Projects/im_19.png",pred_onx[0,:,:,19]*255)
out = pred_onx[0].transpose(2,0,1)*255
cv2.imwrite("/briefcam/Projects/background.png",out[14])
cv2.imwrite("/briefcam/Projects/max.png",np.amax(np.absolute(out[0:15]), axis=0))
tmp2_odd = np.amax(np.absolute(out[15::2]), axis=0)
tmp2_even = np.amax(np.absolute(out[16::2]), axis=0)
for i in range(41):
	cv2.imwrite("/briefcam/Projects/hm_%d.png"%i,np.absolute(out[i]))
cv2.imwrite("/briefcam/Projects/max_x.png",tmp2_odd)
cv2.imwrite("/briefcam/Projects/max_y.png",tmp2_even)