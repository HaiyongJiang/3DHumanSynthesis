# Learning from Synthetic Humans (SURREAL)

This is the code adopted from the following paper:

Gül Varol, Javier Romero, Xavier Martin, Naureen Mahmood, Michael J. Black, Ivan Laptev and Cordelia Schmid, [Learning from Synthetic Humans](https://arxiv.org/abs/1701.01370), CVPR 2017.

Check the [project page](http://www.di.ens.fr/willow/research/surreal/) for more materials.

Contact: [Gül Varol](http://www.di.ens.fr/~varol/).
## Contents
* [1. Create your own synthetic data](https://github.com/gulvarol/surreal#2-create-your-own-synthetic-data)
* [Citation](https://github.com/gulvarol/surreal#citation)
* [License](https://github.com/gulvarol/surreal#license)
* [Acknowledgements](https://github.com/gulvarol/surreal#acknowledgements)


## 1. Create your own synthetic data
### 1.1. Preparation
#### 1.1.1. SMPL data

a) You need to download SMPL for MAYA from http://smpl.is.tue.mpg.de in order to run the synthetic data generation code. Once you agree on SMPL license terms and have access to downloads, you will have the following two files:

This contains the basic smpl model, including the articulated shape, and smpl blending shapes.
```
basicModel_f_lbs_10_207_0_v1.0.2.fbx
basicModel_m_lbs_10_207_0_v1.0.2.fbx
```

Place these two files under `datageneration/smpl_data` folder.

b) With the same credentials as with the SURREAL dataset, you can download the remaining necessary SMPL data and place it in `datageneration/smpl_data`. This includes the motion capture data (MoCap), including the pelvis position (trans) and the poses (poses) for a list of sequences. There are also regression_verts, and joint_regressor to build the connection with joints. Another important information is the shapes (maleshapes, femaleshapes) from real persons. 

``` shell
./download_smpl_data.sh /path/to/smpl_data yourusername yourpassword
```

``` shell
smpl_data/
------------- textures/ # folder containing clothing images (also available at lsh.paris.inria.fr/SURREAL/smpl_data/textures.tar.gz)
------------- (fe)male_beta_stds.npy
------------- smpl_data.npz # 2.5GB
 # trans*           [T x 3]     - (T: number of frames in MoCap sequence)
 # pose*            [T x 72]    - SMPL pose parameters (T: number of frames in MoCap sequence)
 # maleshapes       [1700 x 10] - SMPL shape parameters for 1700 male scans
 # femaleshapes     [2103 x 10] - SMPL shape parameters for 2103 female scans 
 # regression_verts [232]
 # joint_regressor  [24 x 232]
```

*Note: SMPL pose parameters are [MoSh](http://mosh.is.tue.mpg.de/)'ed from CMU MoCap data. Note that these are not the most recent MoSh results. For any questions regarding MoSh, please contact mosh@tue.mpg.de instead. Here, we only provide the pose parameters for MoCap sequences, not their shape parameters (they are not used in this work, we randomly sample body shapes).*

#### 2.1.2. Background images

We only provide names of the background images we used. They are downloaded from [LSUN dataset](http://lsun.cs.princeton.edu/2016/index.html) using [this code](https://github.com/fyu/lsun). You can download images from this dataset or use any other images.

#### 2.1.3. Blender
You need to download [Blender](http://download.blender.org/release/) and install scipy package to run the first part of the code. The provided code was tested with [Blender2.78](http://download.blender.org/release/Blender2.78/blender-2.78a-linux-glibc211-x86_64.tar.bz2), which is shipped with its own python executable as well as distutils package. Therefore, it is sufficient to do the following:

``` shell
# Install pip
/blenderpath/2.78/python/bin/python3.5m get-pip.py
# Install scipy
/blenderpath/2.78/python/bin/python3.5m pip install scipy
```

`get-pip.py` is downloaded from [pip](https://pip.pypa.io/en/stable/installing/). Replace the `blenderpath` with your own and set `BLENDER_PATH`.

Otherwise, you might need to point to your system installation of python, but be prepared for unexpected surprises due to version mismatches. There may not be support about questions regarding this installation.

#### 2.1.4. FFMPEG
If you want to save the rendered images as videos, you will need [ffmpeg](https://ffmpeg.org/) library. Build it and set the `FFMPEG_PATH` to the directory that contains `lib/` and `bin/` folders. Additionally, if you want to use H.264 codec as it is done in the current version of the code, you need to have the [x264](http://www.videolan.org/developers/x264.html) libraries compiled. In that case, set `X264_PATH` to your build. If you use another codec, you don't need `X264_PATH` variable and you can remove `-c:v h264` from `main_part1.py`.

This is how the ffmpeg was built:

``` shell
# x264
./configure  --prefix=/home/gvarol/tools/ffmpeg/x264_build --enable-static --enable-shared --disable-asm
make 
make install

# ffmpeg
./configure --prefix=/home/gvarol/tools/ffmpeg/ffmpeg_build_sequoia_h264 --enable-avresample --enable-pic --disable-doc --disable-static --enable-shared --enable-gpl --enable-nonfree --enable-postproc --enable-x11grab --disable-yasm --enable-libx264 --extra-ldflags="-I/home/gvarol/tools/ffmpeg/x264_build/include -L/home/gvarol/tools/ffmpeg/x264_build/lib" --extra-cflags="-I/home/gvarol/tools/ffmpeg/x264_build/include"
make
make install
```

#### 2.1.5. OpenEXR
The file type for some of the temporary outputs from Blender will be EXR images. In order to read these images, the code uses [OpenEXR bindings for Python](http://www.excamera.com/sphinx/articles-openexr.html). These bindings are available for python 2, the second part of the code (`main_part2.py`) needs this library.

### 2.2. Running the code
Copy the `config.copy` into `config` and edit the `bg_path`, `tmp_path`, `output_path` and `openexr_py2_path` with your own paths.

* `bg_path` contains background images and two files `train_img.txt` and `test_img.txt`. The ones used for SURREAL dataset can be found in `datageneration/misc/LSUN`. Note that the folder structure is flattened for each room type.

* `tmp_path` stores temporary outputs and is deleted afterwards. You can use this for debugging.

* `output_path` is the directory where we store all the final outputs of the rendering.

* `openexr_py2_path` is the path to libraries for [OpenEXR bindings for Python](http://www.excamera.com/sphinx/articles-openexr.html).

`run.sh` script is ran for each clip. You need to set `FFMPEG_PATH`, `X264_PATH` (optional), `PYTHON2_PATH`, and `BLENDER_PATH` variables. `-t 1` option can be removed to run on multi cores, it runs faster.

 ``` shell
# When you are ready, type:
./run.sh
```


## Citation
If you use this code, please cite the following:
> @article{varol17a,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TITLE = {Learning from Synthetic Humans},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AUTHOR = {Varol, G{\"u}l and Romero, Javier and Martin, Xavier and Mahmood, Naureen and Black, Michael J. and Laptev, Ivan and Schmid, Cordelia},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;JOURNAL =  {CVPR},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;YEAR = {2017}  
}

## License
Please check the [license terms](https://github.com/gulvarol/surreal/blob/master/LICENSE.md) before downloading and/or using the code, the models and the data. http://www.di.ens.fr/willow/research/surreal/data/license.html

## Acknowledgements
The data generation code is built by [Javier Romero](https://github.com/libicocco/), [Gul Varol](https://github.com/gulvarol) and [Xavier Martin](https://github.com/martin-xavier).

The training code is written by [Gul Varol](https://github.com/gulvarol) and is largely built on the ImageNet training example [https://github.com/soumith/imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch) by [Soumith Chintala](https://github.com/soumith/).

