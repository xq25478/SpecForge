set -ex

script_dir=$(cd "$(dirname "$0")" && pwd)

pip install --upgrade pip setuptools wheel
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --use-pep517 ninja
pip install jinja2==3.1.2

pushd $script_dir/../../
# pip install -r requirements.txt # 默认环境已装好 
pip install -e .
popd

# 全量编译 安装 flash-attn
# MAX_JOBS=$(nproc) pip install flash-attn --no-build-isolation --no-cache-dir --force-reinstall # 默认环境已装好 

# 全量编译 安装 deepspeed
# MAX_JOBS=$(nproc) \
# TORCH_CUDA_ARCH_LIST="9.0" \
# DS_BUILD_OP=1 \
# TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions \
# pip install deepspeed --force-reinstall --global-option="build_ext" --global-option="-j192"