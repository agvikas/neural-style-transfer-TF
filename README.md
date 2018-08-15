# neural-style-transfer-TF
Implementation of neural style transfer in Tensorflow.

## Requirements
All requirements can be installed by running ```pip install -r requirements.txt ```. If you want to manually install, the list is given below.  
  
- tensorflow / tensorflow-gpu  
- numpy  
- opencv  

## VGG weights
Pretrained VGG-16 weights can be downloaded from [here](https://drive.google.com/open?id=1Sho_UN8SnKRVy20B-B2BXqAN6e9tvTPS).

## How to run?
```python main.py --weights_path <path_to_vgg_weights> --content_image <path_to_content_image> --style_image <path_to_style_image> ```  
or  
```python main.py -wp <path_to_vgg_weights> -ci <path_to_content_image> -si <path_to_style_image> ```  

#### Optional arguments:

| Argument      | Short-hand  | Description  |
| ------------- |:------------|:------------|
| output_path   | op          | Path to save the output file. Default is "art_image.bmp" saved to current working directory. |
| learning_rate | lr          | Learning rate for Adam optimizer. Deafult is 2.                                          |
| iterations    | i           | Number of iterations to run. Default is 2000.                                            |
| alpha         | a           | content loss weight. Default is 100.                                                     |         
| beta          | b           | style loss weight. Default is 8.                                                         |
| layer_loss_weights| lw      | weights for style loss at different layers. Default is [0.5, 1, 0.5, 0.5, 0.5].          |       

# Sample Outputs 
The output images were generated using the following hyperparameters:  
- learning_rate: 2 
- iterations: 5000
- alpha: 100
- beta: 0.5 
- layer_loss_weights: [0.5, 1, 1, 3, 4]

Content image           |  Style image | Output image |
:-------------------------:|:-------------------------:|:----------------------:|
<img src="https://user-images.githubusercontent.com/38666732/44138918-1b104800-a093-11e8-88a1-0e322e3ea765.jpg" width="288"> |<img src="https://user-images.githubusercontent.com/38666732/44139589-181d9b32-a095-11e8-8a0d-4e4376ebd3a2.jpg" width="240">  |  <img src="https://user-images.githubusercontent.com/38666732/44139576-14371570-a095-11e8-9073-fd73b11baa40.png" width="288"> |
<img src="https://user-images.githubusercontent.com/38666732/44139678-6752e072-a095-11e8-84ed-184d614e19e2.jpg" width="260"> |<img src="https://user-images.githubusercontent.com/38666732/44139679-68ea27d8-a095-11e8-98c2-bb7c33db7039.jpg" width="288">  |  <img src="https://user-images.githubusercontent.com/38666732/44139688-6f06caa4-a095-11e8-98aa-22c7b71c24b7.png" width="288"> |
<img src="https://user-images.githubusercontent.com/38666732/44140316-52457c2e-a097-11e8-8fa8-19ee15eaafdf.JPG" width="260"> |<img src="https://user-images.githubusercontent.com/38666732/44140318-54894e84-a097-11e8-8ba4-0f26ccd3806d.jpg" width="288">  |  <img src="https://user-images.githubusercontent.com/38666732/44140341-689a279a-a097-11e8-88d0-29e0f28627cc.png" width="288"> |

