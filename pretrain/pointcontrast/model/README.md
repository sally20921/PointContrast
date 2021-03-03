# Sparse Tensors and Convolutions
* The 3D data of interest in this work consists of 3D scans of surfaces.
* In such datasets, most of the 3D space is empty.
* To handle this sparsity, we use sparse tensors: high-dimensional equivalents of sparse matrices.
* Mathematically, we can represent a sparse tensor for 3D data as a set of coordinates *C* and associated features *F*.
* Convolutions on sparse tensors (also known as sparse convolutions) require somewhat different definition from conventional (dense) convolutions.
* In discrete dense 3D convolution, we extract input features and multiply with a dense kernel matrix.

# Sparse fully-convolutional features
* Fully-convolutional networks consist purely of translation-invariant operations, such as convolutions and elementwise nonlinearities. 
* If we apply a sparse convolutional network to a sparse tensor, we get a sparse output tensor. 
* We refer to the contents of this output tensor as fully-convolutional features.
* We use a UNet structure with skip connections and residual blocks to extract such sparse fully-convolutional features.


