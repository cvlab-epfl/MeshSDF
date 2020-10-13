pip --no-cache-dir install \
	numpy \
	plotly \
  torch \
  torchvision \
	trimesh \
	'git+https://github.com/facebookresearch/pytorch3d.git'

cd external
git clone https://github.com/daniilidis-group/neural_renderer.git
cd neural_renderer && python setup.py install --user && cd ../
