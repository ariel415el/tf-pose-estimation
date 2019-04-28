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
OUT_LAYER=Openpose/concat_stage7
#OUT_LAYER=MobilenetV1/Conv2d_1_depthwise/depthwise
echo ++++++++++++++++++++++++++++++++++++++
echo ++++++++++ Create dir ++++++++++++++++
echo ++++++++++++++++++++++++++++++++++++++

mkdir "${NEW_DIR}"
cp "${MODEL_PATH}/${MODEL}"* "${NEW_DIR}"


echo ++++++++++++++++++++++++++++++++++++++
echo ++++++ Test checkpoint  ++++++++++++++
echo ++++++++++++++++++++++++++++++++++++++

#python3 "${TF_DIR}"/tf_pose/train_like_inference.py --checkpoint "${NEW_DIR}/${MODEL}"  --out_path "${NEW_DIR}"/heatMaps --model_name "${MODEL_TYPE}"
#exit 0
echo ++++++++++++++++++++++++++++++++++++++
echo ++++++ Load checkpoint input +++++++++
echo ++++++++++++++++++++++++++++++++++++++

python3 "${TF_DIR}"/tools/run_checkpoint.py --model "${MODEL_TYPE}" --ckp "${NEW_DIR}/${MODEL}" --name "${NEW_DIR}/${MODEL}"-def --resize "${WIDTH}"x"${HEIGHT}"  

echo ++++++++++++++++++++++++++++++++++++++
echo +++++++++ Freeze model +++++++++++++++
echo ++++++++++++++++++++++++++++++++++++++
#  --input_checkpoint="${NEW_DIR}/${MODEL}" \
#  --input_meta_graph="${NEW_DIR}/${MODEL}".meta \
#  --input_checkpoint="${NEW_DIR}"/generated_checkpoint-1 \
#python3  "${TF_DIR}"/trt/freeze_ariel.py "${NEW_DIR}/${MODEL}"

MODEL_NAME="${MODEL}"_frozen
python3 -m tensorflow.python.tools.freeze_graph \
  --input_graph="${NEW_DIR}/${MODEL}"-def.pb \
  --output_graph="${NEW_DIR}/${MODEL_NAME}".pb \
  --input_checkpoint="${NEW_DIR}/${MODEL}" \
  --output_node_names=""${OUT_LAYER}""

echo ++++++++++++++++++++++++++++++++++++++
echo ++++++++++ Optimize model  +++++++++++
echo ++++++++++++++++++++++++++++++++++++++

NEW_MODEL_NAME="${MODEL_NAME}"_opt
python3 -m tensorflow.python.tools.optimize_for_inference \
    --input="${NEW_DIR}/${MODEL_NAME}".pb \
    --output="${NEW_DIR}/${NEW_MODEL_NAME}".pb  \
    --input_names=image \
    --output_names="${OUT_LAYER}" \
    --transforms='
     strip_unused_nodes(type=float, shape="1,"${HEIGHT}","${WIDTH}",3")
     remove_nodes(op=Identity, op=CheckNumerics)
     fold_constants(ignoreError=False)
     fold_old_batch_norms
     fold_batch_norms'

echo ++++++++++++++++++++++++++++++++++++++
echo ++++++ Make constant input  ++++++++++
echo ++++++++++++++++++++++++++++++++++++++
MODEL_NAME="${NEW_MODEL_NAME}"
NEW_MODEL_NAME="${MODEL_NAME}"_constant_"${WIDTH}"x"${HEIGHT}"
python3 "${TF_DIR}"/tools/make_constant_input.py "${NEW_DIR}/${MODEL_NAME}".pb  "${HEIGHT}" "${WIDTH}"

echo ++++++++++++++++++++++++++++++++++++++
echo ++++++++++ Test model  +++++++++++++++
echo ++++++++++++++++++++++++++++++++++++++
MODEL_NAME="${NEW_MODEL_NAME}"
IMAGE_DIR="${TF_DIR}"/images
python3 "${TF_DIR}"/ariel_run.py --images "${IMAGE_DIR}" --model "${NEW_DIR}/${MODEL_NAME}".pb  --resize "${WIDTH}"x"${HEIGHT}" --in_name image:0 --out_name "${OUT_LAYER}":0 --debug_images
mv "${TF_DIR}"/tf-openpose_"${MODEL_NAME}".json "${NEW_DIR}"/tf-openpose_"${MODEL_NAME}".json
# python3 "${TF_DIR}"/tools/vis/create_debug_images.py  "${TF_DIR}"/images "${NEW_DIR}"/tf-openpose_"${MODEL_NAME}".json
mv  "${IMAGE_DIR}"_out_"${MODEL_NAME}" "${NEW_DIR}"


echo ++++++++++++++++++++++++++++++++++++++
echo ++++++++++ Convert to onnx +++++++++++
echo ++++++++++++++++++++++++++++++++++++++

################## convert to onnx #################
python3 -m tf2onnx.convert --input "${NEW_DIR}/${MODEL_NAME}".pb --inputs image:0 --outputs "${OUT_LAYER}":0 --verbose --output "${NEW_DIR}/${NEW_MODEL_NAME}".onnx



echo ++++++++++++++++++++++++++++++++++++++
echo ++++++ Visualize onnx model+++++++++++
echo ++++++++++++++++++++++++++++++++++++++

################## visualize onnx model  #################

python "${TF_DIR}"/onnx/onnx/tools/net_drawer.py --input "${NEW_DIR}/${NEW_MODEL_NAME}".onnx  --output  "${NEW_DIR}/${NEW_MODEL_NAME}".dot --embed_docstring
dot -Tsvg  "${NEW_DIR}/${NEW_MODEL_NAME}".dot -o   "${NEW_DIR}/${NEW_MODEL_NAME}".svg

