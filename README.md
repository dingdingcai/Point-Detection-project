# Point Detection

This project is for detecting the products which have been taken or moved from the original positions by comparing the  deep features od two target images.
The inputs for this model are a pair of images (A_image and B_image) and the heatmap (A_point) of products within A_image. The output is (Prediction), a heatmap of products changed in A_image related to B_image. The negative example is shown as below:
![Image text](https://github.com/dingdingcai/Point-Detection-project/blob/master/example.png)
A_changed is the ground truth, and the Error is the difference between prediction and ground truth.
