# Distribution Shift Detection for the Postmarket Surveillance of Medical AI Algorithms

This is the code used in the paper "Distribution Shift Detection for the Postmarket Surveillance of Medical AI Algorithms: A Retrospective Simulation Study on Retinal Fundus Images" (link coming soon).

# Prerequisites

* **Python environment**

  ````
  > pip install -e .
  ````


* **Data** 

  The Eyepacs dataset is not publicly available. Enquiries about data access may be directed to contact@eyepacs.org. In the meantime, please use your own dataset for experiments.

# Experiments

## Configurations

All experiments are fully specified by config files which can be found in `./config`. Please adjust paths in there as needed.



## Usage

For examples on how to train and evaluate models, as well as instructions for full reproduction of the paper experiments, check 

```
./scripts/dispatch_experiments.sh
```

Please note the upcoming refactoring, which will make the code easier to adapt to your own environments.


# Citation

If you use this code, please cite

````
@article{koch2023subgroup,
  title      = {Distribution Shift Detection for the Postmarket Surveillance of Medical AI Algorithms: A Retrospective Simulation Study on Retinal Fundus Images},
  author     = {Koch, Lisa M and Baumgartner, Christian F and Berens, Philipp},
  journal    = {coming soon},
  year       = {2023},
}
````

Please also note that code segments from related work were used. If use them, please also cite:

````
@inproceedings{liu2020deepkernel,
  title      = {Learning {Deep} {Kernels} for {Non}-{Parametric} {Two}-{Sample} {Tests}},
  author     = {Liu, Feng and Xu, Wenkai and Lu, Jie and Zhang, Guangquan and Gretton, Arthur and Sutherland, Danica J},
  booktitle  = {Proc. International Conference on Machine Learning (ICML)},
  year       = {2020},
}


````

