# Perspective-Correction
Feature match method please refer to this github. [here](https://github.com/zju3dv/LoFTR)

<div align = center>
<img src="illustration1.png" alt="Cover" width="80%"/> 
</div>




## Requirements
```ruby
cuda: 11.0  
python: 3.6.9  
pytorch: 1.7.0  
torchvision: 0.8.1 
```

## Installation
```ruby
git clone https://github.com/f64051041/Perspective-Correction.git
cd Perspective-Correction/LoFTR/demo 
```

## Quick start
Download weight in `Perspective-Correction/LoFTR/demo`:  [here](https://drive.google.com/file/d/1L6S3X5xSk3c-TMkDiQ4A1A-LsLjyvwm0/view?usp=sharing)
```ruby
python demo_loftr.py --weight last.ckpt --input img --input1 img img/sample1.jpg --input2 img/sample2.jpg 
```
