# Development of a deep learning system in detecting corneal diseases from low-quality slit lamp images
### create time: 2021.09.10

# Introduction
This repository contains the source code for developing a deep learning system  in detecting corneal (DC) diseases from low-quality slit lamp images 
The performance of a deep learning system in detecting corneal diseases from low-quality images can be improved if low-quality images with sufficient diagnostic certainty are added to the training set.

# Prerequisites

* Ubuntu: 18.04 lts
* Python 3.7.8
* Pytorch 1.6.0
* NVIDIA GPU + CUDA_10.0 CuDNN_7.5
This repository has been tested on NVIDIA RTX2080Ti. Configurations (e.g batch size, image patch size) may need to be changed on different platforms.

# Installation
Other packages are as follows:

* wheel
* yaml
* scipy
* joblib
* opencv-python
* scikit-image
* numpy
# Install dependencies:

* pip install -r requirements.txt
# Usage
## Dataset form
<pre>
|-data  
&ensp;&ensp;&ensp;&ensp;|-train  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;|--label1  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;|--*.jpg  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;|--label2  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;|--*.jpg  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;|--label3      
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;|--*.jpg  
&ensp;&ensp;&ensp;&ensp;|-val   
&ensp;&ensp;&ensp;&ensp; ...  
&ensp;&ensp;&ensp;&ensp;|-test   
&ensp;&ensp;&ensp;&ensp;</pre>  

...  
 
## training and testing ##
The training and testing are executed as follows:

* The file "CDtrainingv1.py" in /CD-system-Source is used for our models training.
> python3 CDtrainingv1.py --data PATH

* The file "CDtestingv1.py" in /CD-system-Source is used for testing.
> python3 CDtestingv1.py 


**Please feel free to contact us for any questions or comments: Zhongwen Li, E-mail: li.zhw@qq.com or Jiewei Jiang, E-mail: jiangjw924@126.com.**
