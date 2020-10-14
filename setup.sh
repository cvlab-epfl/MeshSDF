pip install --upgrade pip
pip --no-cache-dir install \
	numpy \
	plotly \
  torch \
  torchvision \
	trimesh \
	plyfile \
	psutil \
	imageio \
	--user

mkdir external && cd external
git clone https://github.com/daniilidis-group/neural_renderer.git && cd neural_renderer
git fetch origin pull/5cc85740240226b6db9e48d5e5089b4e41681a2c/head:torch15
git checkout torch15
python setup.py install --user && cd ../../
