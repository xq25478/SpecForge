set -ex

script_dir=$(cd "$(dirname "$0")" && pwd)

pushd $script_dir/../../
pip install -r requirements.txt
pip install -e .
popd

pip install jinja2==3.1.2

# if use 
# pip install sglang[all] vllm