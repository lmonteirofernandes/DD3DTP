<div id="top"></div>

<h3 align="center">Data-driven surrogate model for predicting the toughness of 3D polycrystals</h3>

  <p align="center">
    This repository contains the code for a data-driven neural network for surrogate modeling brittle fracture of 3D polycrystalline materials. Once trained, a speedup of 150 times can be achieved compared to FFT computations.
    <br />
    <br />
  </p>
</div>

### Built with
Our released implementation is tested on:
* Ubuntu 20.04
* Python 3.8.18
* PyTorch 2.0.1
* NVIDIA CUDA 11.7


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting started

### Prerequisites

* Clone the repository
* Create and launch a conda environment with
  ```sh
  conda create -n DD3DTP python=3.8.18
  conda activate DD3DTP
  ```
<!--### Installation-->
* Install dependencies
    ```sh
  pip install -r requirements.txt
  ```
  Note: for Pytorch CUDA installation follow https://pytorch.org/get-started/locally/.
  
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Datasets
The datasets used in our experiments can be downloaded [here](https://cloud.minesparis.psl.eu/index.php/s/uiTyztVpIZlbMmP) and should be placed inside the data folder.

### Training
To train a model you can use the `train.py` script provided.

### Testing 
The trained models can be tested with the `statistical_analysis.py` script and the `3D_plots.ipynb` notebook.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


### Citation

Please cite our work with
```sh
@article{monteiro_fernandes_data-driven_2025,
	title = {Data-driven surrogate model for predicting the toughness of {3D} polycrystals},
	volume = { },
	issn = { },
	doi = { },
	journal = { },
	author = {Monteiro Fernandes, Lucas and Basso Della Mea, Guilherme and Blusseau, Samy and Rieder, Philipp and Neumann, Matthias and Schmidt, Volker and Proudhon, Henry and Willot, François},
	month =  ,
	year = {2025},
	pages = { },
}
  ```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Lucas Monteiro Fernandes - https://www.linkedin.com/in/lucas-monteiro-fernandes-96b621171 - lucas.monteiro_fernandes@minesparis.psl.eu - lucasmon10@hotmail.com

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This project has received funding from the French Agence Nationale de la Recherche (ANR, ANR-21-FAI1-0003) and the Bundesministerium für Bildung und Forschung (BMBF, 01IS21091) within the French-German research project SMILE.

<p align="right">(<a href="#top">back to top</a>)</p>
