# Use specific version of nvidia cuda image
FROM dese251/sviwan22:run AS runtime

# hf_xet ya viene incluido por defecto con huggingface_hub (reemplaza a hf_transfer,
# que fue eliminado en huggingface_hub v1.0). Evita el timeout de 30 min que tenías con wget.
RUN pip install -U huggingface_hub runpod websocket-client

WORKDIR /

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-WanVideoWrapper && pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess && \
    cd ComfyUI-WanAnimatePreprocess && pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-segment-anything-2 && \
    git clone https://github.com/eddyhhlure1Eddy/IntelligentVRAMNode && \
    git clone https://github.com/eddyhhlure1Eddy/auto_wan2.2animate_freamtowindow_server && \
    git clone https://github.com/eddyhhlure1Eddy/ComfyUI-AdaptiveWindowSize && \
    cd ComfyUI-AdaptiveWindowSize/ComfyUI-AdaptiveWindowSize && \
    mv * ../

RUN mkdir -p /ComfyUI/models/vae /ComfyUI/models/clip_vision /ComfyUI/models/text_encoders \
    /ComfyUI/models/diffusion_models /ComfyUI/models/loras /ComfyUI/models/detection

# Xet en modo alto rendimiento: satura ancho de banda y CPU disponibles para bajar mas rapido
ENV HF_XET_HIGH_PERFORMANCE=1
# Log mas limpio en build
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

# -----------------------------------------------------------------------
# TODAS las descargas en UNA sola capa (RUN), y cada archivo se mueve y
# se limpia la cache inmediatamente despues de bajarlo. Asi nunca queda
# guardado dos veces (cache + local_dir) ni se acumula entre capas.
# -----------------------------------------------------------------------
RUN python3 - <<'EOF'
import os, shutil
from huggingface_hub import hf_hub_download

CACHE = "/tmp/hf_cache"
M = "/ComfyUI/models"

def get(repo_id, filename, dest_path, repo_type="model"):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    # Solo cache_dir (NO local_dir) -> nunca se duplica el archivo
    path = hf_hub_download(repo_id=repo_id, filename=filename,
                            repo_type=repo_type, cache_dir=CACHE)
    real = os.path.realpath(path)     # el blob real, no el symlink
    shutil.move(real, dest_path)      # mv = instantaneo, no copia
    shutil.rmtree(CACHE, ignore_errors=True)  # borra restos (symlinks/snapshots)
    print("OK ->", dest_path, flush=True)

get("Kijai/WanVideo_comfy",
    "Wan2_1_VAE_bf16.safetensors",
    f"{M}/vae/Wan2_1_VAE_bf16.safetensors")

get("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
    "split_files/clip_vision/clip_vision_h.safetensors",
    f"{M}/clip_vision/clip_vision_h.safetensors")

get("Kijai/WanVideo_comfy",
    "umt5-xxl-enc-bf16.safetensors",
    f"{M}/text_encoders/umt5-xxl-enc-bf16.safetensors")


get("Kijai/WanVideo_comfy",
    "LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors",
    f"{M}/loras/WanAnimate_relight_lora_fp16.safetensors")

get("hijdese2020/wan22_datalora",
    "allnsfw/wan22-k3nk4llinon3-15epoc-full-low-k3nk.safetensors",
    f"{M}/loras/wan22-k3nk4llinon3-15epoc-full-low-k3nk.safetensors",
    repo_type="dataset")

get("Wan-AI/Wan2.2-Animate-14B",
    "process_checkpoint/det/yolov10m.onnx",
    f"{M}/detection/yolov10m.onnx")

get("Kijai/vitpose_comfy",
    "onnx/vitpose_h_wholebody_model.onnx",
    f"{M}/detection/vitpose_h_wholebody_model.onnx")

get("Kijai/vitpose_comfy",
    "onnx/vitpose_h_wholebody_data.bin",
    f"{M}/detection/vitpose_h_wholebody_data.bin")

repo_id = "eddy1111111/lightx2v_it2v_adaptive_fusionv_1.safetensors"
for f in [
    "lightx2v_elite_it2v_animate_face.safetensors",
    "WAN22_MoCap_fullbodyCOPY_ED.safetensors",
    "FullDynamic_Ultimate_Fusion_Elite.safetensors",
    "Wan2.2-Fun-A14B-InP-Fusion-Elite.safetensors",
]:
    get(repo_id, f, f"{M}/loras/{f}")

EOF

RUN python3 - <<'EOF'
import os, shutil
from huggingface_hub import hf_hub_download

CACHE = "/tmp/hf_cache"
M = "/ComfyUI/models"

def get(repo_id, filename, dest_path, repo_type="model"):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    # Solo cache_dir (NO local_dir) -> nunca se duplica el archivo
    path = hf_hub_download(repo_id=repo_id, filename=filename,
                            repo_type=repo_type, cache_dir=CACHE)
    real = os.path.realpath(path)     # el blob real, no el symlink
    shutil.move(real, dest_path)      # mv = instantaneo, no copia
    shutil.rmtree(CACHE, ignore_errors=True)  # borra restos (symlinks/snapshots)
    print("OK ->", dest_path, flush=True)


get("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
    "split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors",
    f"{M}/diffusion_models/wan2.2_animate_14B_bf16.safetensors")


EOF

# Limpia caches de pip para ahorrar algo más de espacio en la imagen final
RUN pip cache purge && rm -rf /root/.cache

COPY . .
RUN mkdir -p /ComfyUI/user/default/ComfyUI-Manager
COPY config.ini /ComfyUI/user/default/ComfyUI-Manager/config.ini
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
