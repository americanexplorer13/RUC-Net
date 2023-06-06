# RUC-Net: A Residual-Unet-Based Convolutional Neural Network for Pixel-Level Pavement Crack Segmentation
This is my own PyTorch implementation of 2023 U-Net (RUC-Net) used for crack segmentation. https://www.mdpi.com/1424-8220/23/1/53
# Architecture of RUC-Net
![image](https://github.com/americanexplorer13/rucnet/assets/57260643/485f078c-f300-4d9b-a19a-dc2c70460152)
![image](https://github.com/americanexplorer13/rucnet/assets/57260643/829e2220-9ed7-4063-a9c8-dfbdd8bb181d)
# Usage
pytorch == 2.0.0
# Experiments 
![image](https://github.com/americanexplorer13/rucnet/assets/57260643/de158b12-2a2b-4178-82d0-70d0f618fe0d)
# Bias and limitations
If you would look through my code, you will notice 3 differents from original paperwork: 
- Last conv layer were changed from conv3x3 to conv1x1. I suppose, it's a paperwork typo. 
- GroupNorm instead of BatchNorm. It's preferable to use GN if your batch_size < 32, but if you don't need this, just change it in this code. 
- In my implementation I used scSE block, however in original paperwork there's no restriction to use any of these blocks (SCSE, SSE, CSE) so asap I'll update original scse.py file to maintain these blocks.
