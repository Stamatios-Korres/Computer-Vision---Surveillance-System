# Computer Vision - Surveillance System

In this project we explore building a person search system for surveillance images. It is split into two subproblems:

* Person Re-identification
* Attribute-based Person Search

The goal of the first part  is to retrieve person images in one camera view that are the same identity as a query image in another camera view. Features used are:
**HoG, Color Histogram, LBP, BoW with SIFT**. An [SVM](https://en.wikipedia.org/wiki/Support_vector_machine) classifier has been trained with an RBF kernel. 
The goal of the second part is to retrieve person images given some attributes query attributes (e.g. the gender, whether the person has backpack or bag and so on). This is often built
upon attribute recognition modules (Figure 2) which are functions that inputs a person image and runs a classifier to predict the value of a categorical attribute (e.g., male or female).
Analogous features where used. **Deep Features** were used to enchance performance. 

# Technologies Used

* [Matlab](https://www.mathworks.com/products/matlab.html)
* [Matlab's AlexNet](https://www.mathworks.com/help/deeplearning/ref/alexnet.html)


