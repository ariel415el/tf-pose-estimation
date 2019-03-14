WORK_DIR=/home/host_tf-pose
python -m tf2onnx.convert --input ${WORK_DIR}/models/graph/mobilenet_thin/graph_opt.pb --inputs image:0 --outputs Openpose/concat_stage7:0 --verbose --output ${WORK_DIR}/trt/ariel_model.pb

