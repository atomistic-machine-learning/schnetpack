# Citing SchNetPack

If you use SchNetPack in your research, please cite:

K.T. Schütt, S.S.P. Hessmann, N.W.A. Gebauer, J. Lederer, M. Gastegger.
SchNetPack 2.0: A neural network toolbox for atomistic machine learning.
J. Chem. Phys. 2023, 158 (14): 144801.
[10.1063/5.0138367](https://doi.org/10.1063/5.0138367).

K.T. Schütt, P. Kessel, M. Gastegger, K. Nicoli, A. Tkatchenko, K.-R. Müller.
SchNetPack: A Deep Learning Toolbox For Atomistic Systems.
J. Chem. Theory Comput. 2019, 15 (1): 448-455.
[10.1021/acs.jctc.8b00908](https://doi.org/10.1021/acs.jctc.8b00908).

```bibtex
@article{schutt2023schnetpack,
    author = {Sch{\"u}tt, Kristof T. and Hessmann, Stefaan S. P. and Gebauer, Niklas W. A. and Lederer, Jonas and Gastegger, Michael},
    title = "{SchNetPack 2.0: A neural network toolbox for atomistic machine learning}",
    journal = {The Journal of Chemical Physics},
    volume = {158},
    number = {14},
    pages = {144801},
    year = {2023},
    month = {04},
    issn = {0021-9606},
    doi = {10.1063/5.0138367},
    url = {https://doi.org/10.1063/5.0138367},
    eprint = {https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/5.0138367/16825487/144801\_1\_5.0138367.pdf},
}
@article{schutt2019schnetpack,
    author = {Sch{\"u}tt, Kristof T. and Kessel, Pan and Gastegger, Michael and Nicoli, Kim A. and Tkatchenko, Alexandre and Müller, Klaus-Robert},
    title = "{SchNetPack: A Deep Learning Toolbox For Atomistic Systems}",
    journal = {Journal of Chemical Theory and Computation},
    volume = {15},
    number = {1},
    pages = {448-455},
    year = {2019},
    doi = {10.1021/acs.jctc.8b00908},
    URL = {https://doi.org/10.1021/acs.jctc.8b00908},
    eprint = {https://doi.org/10.1021/acs.jctc.8b00908},
}
```

## Extensions

SchNetPack can be the base for implementations of advanced atomistic neural networks
and training tasks. For example, the extension package
[schnetpack-gschnet](https://github.com/atomistic-machine-learning/schnetpack-gschnet)
implements cG-SchNet, a conditional generative model for molecules. It shows how a
complex training task fits into a few custom classes while using the hierarchical
configuration and automated training of SchNetPack.
