# Cluster Colors of Picture

### Description
   * Using Kmeans to find the means of GMM (Gaussian Mixture Model)
     to cluster colors in a picture, then run GMM algorithm to achieve
     our goal. If you want, you can save the result as a JPEG file.
     
   * **Results:**
   
     _1. origin image:_ 
     
     ![Origin Image](https://github.com/bobolee1239/cluster_colors_of_pic/blob/master/SamplePics/img.jpg)
     
     _2. GMM for 5 clusters:_
     
     ![GMM_5](https://github.com/bobolee1239/cluster_colors_of_pic/blob/master/SamplePics/GMM_5.jpg)
     
     _3. GMM for 20 clusters:_
     
     ![GMM_20](https://github.com/bobolee1239/cluster_colors_of_pic/blob/master/SamplePics/GMM_20.jpg)
     
### Solving  Other  Clustering  Problems
   * If you like to use this structure to deal another cluster problem, 
     you can customize your class to represent your point in GMM process.
     Just inheriant the class Point in 

>         ./src/Kmeans.py 
     
### Things to take care while doing GMM
   * If members in a cluster is too dense, the covariance matrix would become
     extremely singular. Therefore, we should add some isotropic noise into 
     covariance matrix to prevent singularity and make the model robust to noise.
