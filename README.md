# Cluster Colors of Picture

## Description
   * Using Kmeans to find the means of GMM (Gaussian Mixture Model)
     to cluster colors in a picture, then run GMM algorithm to achieve
     our goal. If you want, you can save the result as a JPEG file.
     
   * **Results:**
     _1. origin image:_ 
     
     ![Origin Image](https://drive.google.com/file/d/13S8cV9YfXFRKay0EmnpQ5a84QMON3cnx/view?usp=sharing)
     
     _2. GMM for 5 clusters:_
     
     ![GMM_5](https://drive.google.com/file/d/1gc2aGpdxWU19tLl_v7ygZNbvsmwRhcJt/view?usp=sharing)
     
     _3. GMM for 20 clusters:_
     
     ![GMM_20](https://drive.google.com/file/d/17gQ2Q8RSQqAEGqQYo6_4n47a267rWnRC/view?usp=sharing)
     
## Solving Other Clustering Problems
   * If you like to use this structure to deal another cluster problem, 
     you can customize your class to represent your point in GMM process.
     Just import src/Kmeans.py and inheriant from class Point.
