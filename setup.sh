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
	gdown \
	--user

cd data
gdown https://drive.google.com/file/d/1KCnZjWUuQQSGjc2C_Z0_j4IFBYmZ8GvG/view?usp=sharing
