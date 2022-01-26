Real-CUGAN for VapourSynth, User Guide
-------------------------------------------
simple-upcunet.vpy is a simple demo.<br>
You should append file the root of "upcunet_v3_vs.py" into sys.path<br>
RealWaifuUpScaler init setting：
  - scale: 2x/3x/4x;
  - weight_path：the path of the weights of model;
  - device: cuda device number;
  - tile_mode：0/1/2/3/4. The bigger the number, the less video memory is needed, and the lower inference speed it is.


Tested in (environment):
  - PyTorch==1.10.1+cu111
  - Python==3.9.5
  - VapourSynth==R57

You should install PyTorch using pip, the version of PyTorch should be >=1.0.0 (for 30 series N card: >=1.9.0; A card is not supported)<br>
VapourSynth version: >=R54-API4-test2
