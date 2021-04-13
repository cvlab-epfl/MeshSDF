pip install --upgrade pip
pip --no-cache-dir install \
	numpy \
	plotly \
  torch==1.4.0 \
	trimesh \
	plyfile \
	psutil \
	imageio \
	neural_renderer_pytorch \
	--user

curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
