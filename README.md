# ENVIRONMENT SETUP

## System dependencies

The supported platform for execution of this project is Ubuntu Linux 14.04.  This project has several external dependencies that should be installed through the system package manager:

        sudo apt-get build-dep python-sklearn
        sudo apt-get install gfortran libopenblas-dev liblapack-dev build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev libatlas3gf-base

### Building dependencies from source (optional)

If you encounter any issues with your system packages for external dependencies, you may want to try building them from source.  Note that **this is not the recommended approach** and should only be attempted in case something goes wrong with your distribution packages.

-   [`scikit-learn`](http://scikit-learn.org/stable/install.html)


## Python interpreter

**Use Python 2.7.9** to run this project. Virtualenv or an alternative is recommended.


## Python dependencies

    pip install numpy
    pip install scipy scikit-learn pytz



