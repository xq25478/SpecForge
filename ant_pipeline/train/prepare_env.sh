set -ex

script_dir=$(cd "$(dirname "$0")" && pwd)

pip install --upgrade pip setuptools wheel
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --use-pep517 ninja

pushd $script_dir/../../
# pip install -r requirements.txt # 默认环境已装好 
pip install -e .
popd

# MAX_JOBS=200 pip install flash-attn --no-build-isolation --no-cache-dir --force-reinstall # 默认环境已装好 
pip install jinja2==3.1.2
pip install -U deepspeed