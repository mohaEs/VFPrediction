
Source: <br/>
based on the  model introduced in:
https://doi.org/10.1371/journal.pone.0214875 <br/>
Accessed 20 Jan. 2021: <br/>
https://github.com/uw-biomedical-ml/hvfProgression 


This version of model is **not** exactly the mentioned code in the source. <br/>
The original source code uses Keras. I have modified it to use tensorflow.keras instead of Keras. <br/>
And some other minor changes to make the code works properly with Python 3 and newer version of packages.
Two Pr_Train* scripts are added along with two small fake data for training/testing.<br/>

Notice:  <br/>
In paper's implementation, It is supposed that, the background (non VF values) is -1. <br/>
make sure the background returned by the function FnConvert_1D_VF_2_2D has -1 values. 


