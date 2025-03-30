# Text Binary Detection Paddle
![plot](images/db_net.png)
# requirements
paddlepaddle==3.0.0rc1
# Download weight 
```bash
mkdir weights
cd weights
wget https://github.com/phu2007vis/Paddle_Binary_Detection/releases/download/sdsd/det_r50_icdar15_v5.zip
unzip det_r50_icdar15_v5.zip
cd ..
```
# Replace weight path in tempate.yaml  (best_accuracy instead of best_accuracy.pdparams)
# Test ( result save in visualize.jpg)

![plot](images/visualize.jpg)
