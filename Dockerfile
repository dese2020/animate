# Use specific version of nvidia cuda image
FROM dese251/sviwan22:run AS runtime

#RUN pip install -U "huggingface_hub[hf_transfer]"
RUN pip install huggingface_hub
RUN pip install runpod websocket-client

WORKDIR /

#RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
#    cd /ComfyUI && \
#    pip install -r requirements.txt

#RUN cd /ComfyUI/custom_nodes && \
#    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
#    cd ComfyUI-Manager && \
#    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-WanVideoWrapper && \
    pip install -r requirements.txt

#RUN cd /ComfyUI/custom_nodes && \
#    git clone https://github.com/kijai/ComfyUI-KJNodes && \
#    cd ComfyUI-KJNodes && \
#    pip install -r requirements.txt

#RUN cd /ComfyUI/custom_nodes && \
#    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
#    cd ComfyUI-VideoHelperSuite && \
#    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess && \
    cd ComfyUI-WanAnimatePreprocess && \
    pip install -r requirements.txt
    
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-segment-anything-2 && \
    git clone https://github.com/eddyhhlure1Eddy/IntelligentVRAMNode && \
    git clone https://github.com/eddyhhlure1Eddy/auto_wan2.2animate_freamtowindow_server && \
    git clone https://github.com/eddyhhlure1Eddy/ComfyUI-AdaptiveWindowSize && \
    cd ComfyUI-AdaptiveWindowSize/ComfyUI-AdaptiveWindowSize && \
    mv * ../

#RUN pip install --upgrade onnxruntime-gpu==1.22

# 1. Obliga a Hugging Face a usar una ruta con espacio para su caché
ENV HF_HUB_CACHE=/ComfyUI/models/diffusion_models/.cache

# 2. Obliga a Python (tempfile) a usar una ruta con espacio para los fragmentos temporales
ENV TMPDIR=/ComfyUI/models/diffusion_models/.tmp

# Crea los directorios antes para evitar problemas de permisos
RUN mkdir -p /ComfyUI/models/diffusion_models/.cache /ComfyUI/models/diffusion_models/.tmp

RUN wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors -O /ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors
RUN wget -q https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors -O /ComfyUI/models/clip_vision/clip_vision_h.safetensors
#RUN wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors -O /ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors
#RUN wget -q https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors -O /ComfyUI/models/diffusion_models/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Kijai/WanVideo_comfy', filename='umt5-xxl-enc-bf16.safetensors', local_dir='/ComfyUI/models/text_encoders/', local_dir_use_symlinks=False)"
#RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Kijai/WanVideo_comfy_fp8_scaled', filename='Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors', local_dir='/ComfyUI/models/diffusion_models/', local_dir_use_symlinks=False)"
#RUN mv /ComfyUI/models/diffusion_models/Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors /ComfyUI/models/diffusion_models/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors 
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Comfy-Org/Wan_2.2_ComfyUI_Repackaged', filename='split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors', local_dir='/ComfyUI/models/diffusion_models/', local_dir_use_symlinks=False)"
RUN mv /ComfyUI/models/diffusion_models/split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors /ComfyUI/models/diffusion_models/wan2.2_animate_14B_bf16.safetensors 

#RUN wget -q https://huggingface.co/eddy1111111/lightx2v_it2v_adaptive_fusionv_1.safetensors/resolve/main/lightx2v_elite_it2v_animate_face.safetensors -O /ComfyUI/models/loras/lightx2v_elite_it2v_animate_face.safetensors
#RUN wget -q https://huggingface.co/eddy1111111/lightx2v_it2v_adaptive_fusionv_1.safetensors/resolve/main/WAN22_MoCap_fullbodyCOPY_ED.safetensors -O /ComfyUI/models/loras/WAN22_MoCap_fullbodyCOPY_ED.safetensors
#RUN wget -q https://huggingface.co/eddy1111111/lightx2v_it2v_adaptive_fusionv_1.safetensors/resolve/main/FullDynamic_Ultimate_Fusion_Elite.safetensors -O /ComfyUI/models/loras/FullDynamic_Ultimate_Fusion_Elite.safetensors
#RUN wget -q https://huggingface.co/eddy1111111/lightx2v_it2v_adaptive_fusionv_1.safetensors/resolve/main/Wan2.2-Fun-A14B-InP-Fusion-Elite.safetensors -O /ComfyUI/models/loras/Wan2.2-Fun-A14B-InP-Fusion-Elite.safetensors 
RUN wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors -O /ComfyUI/models/loras/WanAnimate_relight_lora_fp16.safetensors
RUN wget -q https://huggingface.co/datasets/hijdese2020/wan22_datalora/resolve/main/allnsfw/wan22-k3nk4llinon3-15epoc-full-low-k3nk.safetensors -O /ComfyUI/models/loras/wan22-k3nk4llinon3-15epoc-full-low-k3nk.safetensors
RUN mkdir -p /ComfyUI/models/detection

#RUN wget  https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx -O /ComfyUI/models/detection/yolov10m.onnx
#RUN wget  https://huggingface.co/Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_model.onnx -O /ComfyUI/models/detection/vitpose_h_wholebody_model.onnx
#RUN wget  https://huggingface.co/Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_data.bin -O /ComfyUI/models/detection/vitpose_h_wholebody_data.bin

RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Wan-AI/Wan2.2-Animate-14B', filename='process_checkpoint/det/yolov10m.onnx', local_dir='/ComfyUI/models/', local_dir_use_symlinks=False)"
RUN mv /ComfyUI/models/process_checkpoint/det/yolov10m.onnx /ComfyUI/models/detection/yolov10m.onnx 
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Kijai/vitpose_comfy', filename='onnx/vitpose_h_wholebody_model.onnx', local_dir='/ComfyUI/models/', local_dir_use_symlinks=False)"
RUN mv /ComfyUI/models/onnx/vitpose_h_wholebody_model.onnx /ComfyUI/models/detection/vitpose_h_wholebody_model.onnx 
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Kijai/vitpose_comfy', filename='onnx/vitpose_h_wholebody_data.bin', local_dir='/ComfyUI/models/', local_dir_use_symlinks=False)"
RUN mv /ComfyUI/models/onnx/vitpose_h_wholebody_data.bin /ComfyUI/models/detection/vitpose_h_wholebody_data.bin 


RUN python3 - <<'EOF'
import os
from huggingface_hub import hf_hub_download

repo_id = "eddy1111111/lightx2v_it2v_adaptive_fusionv_1.safetensors"
repo_type = "model"
local_dir = "/ComfyUI/models/loras/"

files = [
    "lightx2v_elite_it2v_animate_face.safetensors",
    "WAN22_MoCap_fullbodyCOPY_ED.safetensors",
	"FullDynamic_Ultimate_Fusion_Elite.safetensors",
	"Wan2.2-Fun-A14B-InP-Fusion-Elite.safetensors"
]

for f in files:
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=f,
		local_dir=local_dir
    )

EOF

# Opcional: Limpia la caché temporal para que la imagen de Docker final no pese el doble
RUN rm -rf /ComfyUI/models/diffusion_models/.cache /ComfyUI/models/diffusion_models/.tmp

COPY . .
RUN mkdir -p /ComfyUI/user/default/ComfyUI-Manager
COPY config.ini /ComfyUI/user/default/ComfyUI-Manager/config.ini
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]