# HiDPy

## What is HiDPy?
HiDPy is a pythonic implementation of the HI-D technique that was preseneted earlier by Shaban et al., 2020. 
Compared to the original implementation that was done in MATLAB, this implementation uses Python and a list of open source libraries that provide similar functionality, nevertheless with improved performance. The tools is composed of a set of documents Python notebooks that can be executed by users having limited programming experienc.   

## Dependencies 

1. Scientific computing package: OpenCV (opencv.org) and its Python bindings.
2. Python’s scientific computing package: Numpy (numpy.org).
3. Python’s scientific computing package: Scipy (scipy.org).
4. Python’s plotting package: Matplotlib (matplotlib.org).
5. Python’s plotting package: Seaborn (seaborn.pydata.org).
6. Python’s probabilistic models’ package: Pomegranate (pomegranate.readthedocs.io).
7. Python’s pipelining package: joblib (joblib.readthedocs.io).
8. Python’s concurrency package: multiprocessing (docs.python.org/3/library/multiprocessing.html).
9. Image processing application: Fiji (https://imagej.net/software/fiji). This is an optional package.
10. Jupyter notebook with Visual Studio Code (https://code.visualstudio.com).

To install the dependencies, please run the following command (using python 3.8 or later)

```
pip3 install -r requirements.txt 
```

## Data Sources 

The data used in this study is available from: 
1. [Shaban et al., 2020](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02002-6). The datasets are available from this link on [Zenodo](https://zenodo.org/record/3634348#.Y4DBStLMJGo).
2. [Miron et al., Science Advances 2020](https://www.science.org/doi/10.1126/sciadv.aba8811), Use the following Data Source (IDR0086).

## How to use?
The code can be execlusively used from Python notebooks. A detailed documentation is available in the [Wiki](https://github.com/haitham-shaban/hidpy/wiki) page. 

## Sample datasets
A few sample datasets are availabel in the [data](https://github.com/haitham-shaban/hidpy/tree/main/data/) directory. 