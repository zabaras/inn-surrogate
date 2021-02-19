# Solving inverse problems using conditional invertible neural networks.

Solving inverse problems using conditional invertible neural networks. [JCP](https://www.sciencedirect.com/science/article/pii/S0021999121000899#se0110) [ArXiv](https://arxiv.org/abs/2007.15849)

Govinda Anantha Padmanabha, [Nicholas Zabaras](https://www.zabaras.com/)

PyTorch Implementation of Solving inverse problems using conditional invertible neural networks.

# Highlights
* Rather than developing a surrogate for a forward model, we are training directly an inverse surrogate mapping output information of a physical system to an unknown input distributed parameter.
* A generative model based on conditional invertible neural networks (cINN) is developed.
* The cINN is trained to serve as an inverse surrogate model of physical systems governed by PDEs.
* The inverse surrogate model is used for the solution of inverse problems with unknown spatially-dependent parameters.
* The developed method is applied for the estimation of a non-Gaussian permeability field in multiphase flows using limited pressure and saturation data.

## Inverse surrogate model:
<p align="center">
 <img src="/2D/images/Pic1-3.png" width="300">
 </p> 
 <p align="center">
Mapping: observations &#8594 input space
 </p> 
 
# Dependencies
[PyTorch](https://pytorch.org/) 1.0.0   
Python 3  
[H5py](https://www.h5py.org/)  
[Matplotlib](https://matplotlib.org/stable/index.html)  
[Numpy](https://numpy.org/)  


# Citation  
If you find this GitHub repository useful for your work, please consider to cite this work:  


@article{padmanabha2021solving,     
  title={Solving inverse problems using conditional invertible neural networks},     
  journal={Journal of Computational Physics},     
  pages={110194},     
  year={2021},     
  publisher={Elsevier}     
  doi = {https://doi.org/10.1016/j.jcp.2021.110194 },       
  url = {https://www.sciencedirect.com/science/article/pii/S0021999121000899},       
  author = {Govinda Anantha Padmanabha and Nicholas Zabaras}     
}

