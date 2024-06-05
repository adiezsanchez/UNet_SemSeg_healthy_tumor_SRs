# UNet_SemSeg_healthy_tumor_SRs
Semantic segmentation of healthy vs tumor tissue in mouse intestinal swiss rolls

<h2>Instructions</h2>

1. In order to run these Jupyter notebooks you will need to familiarize yourself with the use of Python virtual environments using Mamba. See instructions [here](https://biapol.github.io/blog/mara_lampert/getting_started_with_mambaforge_and_python/readme.html).

2. Then you will need to create a virtual environment (venv) either using the following command or recreate the environment from the .yml file you can find in the envs folder:

   <code>mamba create --name EMBL_tensorflow python=3.9 devbio-napari csbdeep cudatoolkit=11.2 cudnn=8.1.0 -c conda-forge</code>

3. Afterwards you will need to activate the resulting environment and install a few more packages manually via pip install.

   <code>mamba activate EMBL_tensorflow</code>

   <code>pip install "tensorflow<2.11"</code>

   <code>pip install git+https://github.com/stardist/augmend.git</code>

4. The resulting environment will allow you to run the model training in the GPU, if you want to check that tensorflow has been properly installed you can run:

- To verify CPU config:

<code>python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"</code>

- To verify GPU config (it should return a list of available CUDA devices):

<code>python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"</code>
