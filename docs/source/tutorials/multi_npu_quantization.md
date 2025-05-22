# Multi-NPU (QwQ 32B W8A8)

## Run docker container:
:::{note}
w8a8 quantization feature is supported by v0.8.4rc2 or higher
:::

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-it $IMAGE bash
```

## Install modelslim and convert model
:::{note}
You can choose to convert the model yourself or use the quantized model we uploaded, 
see https://www.modelscope.cn/models/vllm-ascend/QwQ-32B-W8A8
:::

```bash
# (Optional)This tag is recommended and has been verified
git clone https://gitee.com/ascend/msit -b modelslim-VLLM-8.1.RC1.b020

cd msit/msmodelslim
# Install by run this script
bash install.sh
pip install accelerate

cd example/Qwen
# Original weight path, Replace with your local model path
MODEL_PATH=/home/models/QwQ-32B
# Path to save converted weight, Replace with your local path
SAVE_PATH=/home/models/QwQ-32B-w8a8

# In this conversion process, the npu device is not must, you can also set --device_type cpu to have a conversion
python3 quant_qwen.py --model_path $MODEL_PATH --save_directory $SAVE_PATH --calib_file ../common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu --anti_method m1 --trust_remote_code True
```

## Verify the quantized model
The converted model files looks like:
```bash
.
|-- config.json
|-- configuration.json
|-- generation_config.json
|-- quant_model_description.json
|-- quant_model_weight_w8a8.safetensors
|-- README.md
|-- tokenizer.json
`-- tokenizer_config.json
```

Run the following script to start the vLLM server with quantize model:
```bash
vllm serve /home/models/QwQ-32B-w8a8  --tensor-parallel-size 4 --served-model-name "qwq-32b-w8a8" --max-model-len 4096 --quantization ascend
```

Once your server is started, you can query the model with input prompts
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwq-32b-w8a8",
        "prompt": "what is large language model?",
        "max_tokens": "128",
        "top_p": "0.95",
        "top_k": "40",
        "temperature": "0.0"
    }'
```
