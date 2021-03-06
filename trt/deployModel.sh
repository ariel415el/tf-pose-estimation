#!/bin/bash
MODEL_PATH=$1
MODEL=$2
HEIGHT=368 #736
WIDTH=432 #656 #1312
MODEL_TYPE=mobilenet_thin
TF_DIR="$( cd "$(dirname "$0")"/.. ; pwd -P )"
NEW_DIR="${MODEL_PATH}/${MODEL}"
echo workdir: "${TF_DIR}"
echo model path: "${MODEL_PATH}"
echo model name: "${MODEL}"
echo new_dir: "${NEW_DIR}"
OPT=_opt
echo ++++++++++++++++++++++++++++++++++++++
echo ++++++++++ Create dir ++++++++++++++++
echo ++++++++++++++++++++++++++++++++++++++

mkdir "${NEW_DIR}"
cp "${MODEL_PATH}/${MODEL}"* "${NEW_DIR}"


echo ++++++++++++++++++++++++++++++++++++++
echo ++++++ Test checkpoint  ++++++++++++++
echo ++++++++++++++++++++++++++++++++++++++

python3 "${TF_DIR}"/tf_pose/train_like_inference.py --checkpoint "${NEW_DIR}/${MODEL}"  --out_path "${NEW_DIR}"/heatMaps --model_name "${MODEL_TYPE}"

echo ++++++++++++++++++++++++++++++++++++++
echo ++++++ Load checkpoint input +++++++++
echo ++++++++++++++++++++++++++++++++++++++

python3 "${TF_DIR}"/run_checkpoint.py --model "${MODEL_TYPE}" --ckp "${NEW_DIR}/${MODEL}" --name "${MODEL}"-def --width "${WIDTH}" --height "${HEIGHT}"
 

echo ++++++++++++++++++++++++++++++++++++++
echo +++++++++ Freeze model +++++++++++++++
echo ++++++++++++++++++++++++++++++++++++++
#  --input_checkpoint="${NEW_DIR}/${MODEL}" \
#  --input_meta_graph="${NEW_DIR}/${MODEL}".meta \
#  --input_checkpoint="${NEW_DIR}/${MODEL}" \
python3 -m tensorflow.python.tools.freeze_graph \
  --input_graph="${NEW_DIR}/${MODEL}"-def.pb \
  --output_graph="${NEW_DIR}/${MODEL}"_frozen.pb \
  --input_checkpoint="${NEW_DIR}"/generated_checkpoint-1 \
  --output_node_names="Openpose/concat_stage7"
#python3  "${TF_DIR}"/trt/freeze_ariel.py "${NEW_DIR}/${MODEL}"
echo ++++++++++++++++++++++++++++++++++++++
echo ++++++++++ Optimize model  +++++++++++
echo ++++++++++++++++++++++++++++++++++++++

python3 -m tensorflow.python.tools.optimize_for_inference \
    --input="${NEW_DIR}/${MODEL}"_frozen.pb \
    --output="${NEW_DIR}/${MODEL}"_frozen"${OPT}".pb \
    --input_names=image \
    --output_names='Openpose/concat_stage7' \
    --transforms='
     strip_unused_nodes(type=float, shape="1,368,368,3")
     remove_nodes(op=Identity, op=CheckNumerics)
     fold_constants(ignoreError=False)
     fold_old_batch_norms
     fold_batch_norms'

echo ++++++++++++++++++++++++++++++++++++++
echo ++++++ Make constant input  ++++++++++
echo ++++++++++++++++++++++++++++++++++++++

python3 "${TF_DIR}"/trt/make_constant_input.py "${NEW_DIR}/${MODEL}"_frozen"${OPT}".pb  "${HEIGHT}" "${WIDTH}"

echo ++++++++++++++++++++++++++++++++++++++
echo ++++++++++ Test model  +++++++++++++++
echo ++++++++++++++++++++++++++++++++++++++

python3 "${TF_DIR}"/ariel_run.py --images "${TF_DIR}"/images --model "${NEW_DIR}/"${MODEL}_frozen"${OPT}"_constant.pb  --resize "${HEIGHT}"x"${WIDTH}" --in_name const_input:0 --out_name ariel_openpose/Openpose/concat_stage7:0
mv "${TF_DIR}"/tf-openpose_"${MODEL}"_frozen"${OPT}"_constant_"${HEIGHT}"x"${WIDTH}".json "${NEW_DIR}"/tf-openpose_"${MODEL}"_frozen"${OPT}"_constant_"${HEIGHT}"x"${WIDTH}".json
python3 "${TF_DIR}"/vis/create_debug_images.py  "${TF_DIR}"/images "${NEW_DIR}"/tf-openpose_"${MODEL}"_frozen"${OPT}"_constant_"${HEIGHT}"x"${WIDTH}".json
mv  "${TF_DIR}"/images_out_"${MODEL}"_frozen"${OPT}"_constant_"${HEIGHT}"x"${WIDTH}" "${NEW_DIR}"


echo ++++++++++++++++++++++++++++++++++++++
echo ++++++++++ Convert to onnx +++++++++++
echo ++++++++++++++++++++++++++++++++++++++

################## convert to onnx #################
python3 -m tf2onnx.convert --input "${NEW_DIR}/${MODEL}"_frozen"${OPT}"_constant.pb --inputs const_input:0 --outputs ariel_openpose/Openpose/concat_stage7:0 --verbose --output "${NEW_DIR}/${MODEL}"_frozen"${OPT}"_constant.onnx



echo ++++++++++++++++++++++++++++++++++++++
echo ++++++ Visualize onnx model+++++++++++
echo ++++++++++++++++++++++++++++++++++++++

################## visualize onnx model  #################

python "${TF_DIR}"/onnx/onnx/tools/net_drawer.py --input "${NEW_DIR}/${MODEL}"_frozen"${OPT}"_constant.onnx  --output  "${NEW_DIR}/${MODEL}"_frozen"${OPT}"_constant.dot --embed_docstring
dot -Tsvg  "${NEW_DIR}/${MODEL}"_frozen"${OPT}"_constant.dot -o   "${NEW_DIR}/${MODEL}"_frozen"${OPT}"_constant.svg

