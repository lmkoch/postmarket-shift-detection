# Deep Hypothesis Tests Detect Clinically Relevant Subgroup Shifts in Medical Images

This is the code used in the paper "Deep Hypothesis Tests Detect Clinically Relevant Subgroup Shifts in Medical Images" (link coming soon).

![image](https://drive.google.com/file/d/1utYCIwMupnieFA2veuIfE0JZ_QjN2h_o/view?usp=share_link)


# Prerequisites

* **Python environment**

  ````
  > pip install -e .
  ````


* **Data** 

  For both MNIST and Camelyon17, the data is automatically to `data_root` directory specified in config the first time the dataset is created. For Camelyon17, this may take a while.

  The Eyepacs dataset is not publicly available.

# Experiments

## Configurations

All experiments are fully specified by config files which can be found in `./config`. Please adjust paths in there as needed.



## Usage

For examples on how to train and evaluate models, as well as instructions for full reproduction of the paper experiments, check 

```
./scripts/dispatch_experiments.sh
```

Please note the upcoming refactoring, which will make the code easier to adapt your own environments.


# Citation

If you use this code, please cite

````
@article{koch2023subgroup,
  title      = {Deep Hypothesis Tests Detect Clinically Relevant Subgroup Shifts in Medical Images},
  author     = {Koch, Lisa M and Sch{\"u}rch, Christian M and Baumgartner, Christian F and Gretton, Arthur and Berens, Philipp},
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

