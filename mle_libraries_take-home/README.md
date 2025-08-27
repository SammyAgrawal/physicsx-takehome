# pclib: Point Cloud Machine Learning Library
Submission by Samarth Agrawal, August 27th


# Setup 
I created an environment.yml for easy initialization of a conda environment. 
- Can create with conda env create -f environment.yml. 

# Code Structure

Dataset class and data transforms class. 

data transforms handles all the types of data transformations. 

# Approach
When I first read the assignment, my first instinct was to brainstorm really cool processing transformations on 3d point clouds, because that is fun. Imagination stretched to Graph NNs, tree based clustering methods for dimensionality reduction, turning a point cloud into a mesh and back, etc. But then I started exploring the sample notebooks and noted how even methods used internally by trimesh have dependencies not allowed in the take home spec (such as scipy). 

That prompted me to read more carefully and conclude the point of this exercise is less about implementing cool things and more about a reasonable foundation for doing so. Therefore I spent the first 45 minutes instead designing what I think is an extensible interface that, libraries permitting, can easily accomodate cool features with more libraries permitted. 3/4 criteria were architectural thinking, usability, and maintainability after all!

Hence more thought was in setting up the data_transforms and dataset classes and subclasses, and in ensuring that everything worked well in testing. 

An example workflow of a likely processing workflow is in train_and_run_model.ipynb that shows how the pieces could fit together to train a model. Did not extend baseline model. 

## With more time
- Would add unit testing and cases to see if file format is not valid (i.e. load does not work)
- 