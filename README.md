# IsUMap

IsUMap is a dimension reduction and data visualization tool, that can be viewed as a combination of the manifold learning algorithms [UMAP](https://github.com/lmcinnes/umap) and [Isomap](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html).

The theoretical basis of IsUMap is explained in the following publications:
  - P1
  - P2

Please cite them when using IsUMap.

# Usage

Right now, we do not yet provide IsUMap as a package. 
But since it is rather lightweight, you can use it simply by cloning the repo, and installing the dependencies in `environment.yml` (when installing with `conda`) or `requirements.txt` (when installing with `pip`). 

For example, assuming you want to install with `conda` to a new environment, you can run:
```
git clone git@github.com:LUK4S-B/IsUMap.git && cd IsUMap
conda env create -f environment.yml
conda activate isumap
```
In case you want to use `pip`, you can run
```
git clone git@github.com:LUK4S-B/IsUMap.git && cd IsUMap
pip install -r requirements.txt
```

After that, you should be able to run `python minimal_example.py`. Other examples are described in the next section below.

# Examples

