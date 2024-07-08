# whst
Whole Heart Segmentation Transformer

## Content

## Instruction
- data: includes ```mscmr.py``` and ```transforms.py```
	- ```mscmr.py``` is a script to deinfe the method of loading dataset
	- ```transforms.py``` provides the methods of pre-processing data
- logs: a folder to log training information
- models: includes ```headtail.py```, ```position_encoding.py```, ```segmentation.py```, ```transformer.py```, and ```unet_model.py```
	- ```headtail.py``` is a script to define the heads and tails of WHST
	- ```position_encoding.py``` is a script to define the methods of building position encoding
	- ```segmentation.py``` is a script to define the architecture of WHST
	- ```transformer.py``` is a script to define the architecture of widely used transformer.
	- ```unet_model.py``` provides the network architecture of a U-net
- util: provides some functions in ```misc.py```
- ```demo.sh``` provides the examples of training models, such as Unet and WHST
- ```engine.py``` provides the methods of training and evaluating models
- ```main.py``` is a script to define required arguments, load dataset, build models, train models, and evaluate models. 
