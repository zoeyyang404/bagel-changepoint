# Bagel

Bagel detects changepoints in time series using Bayesian linear models. We
can analyse a complete series or update a detector as each new observation arrives.
The R and Python packages use the same C++20 algorithm.

## Start here

We describe R installation [below](#using-bagel-in-r) and Python installation
in the [Python guide and example](python/README.md).
The Python package is called `bagel-changepoint` and imported as `bagel`;
it has not yet been released on PyPI.

The repository is organised as follows:

```text
c++/           Shared algorithm and standalone C++ examples
bagelR/        R package
python/        Python package
tools/         Scripts for copying the shared code into each package
```

## Shared C++ development

We make algorithm changes in `c++/`, then copy them into both packages before
building. The script uses `tools/core-files.txt` to identify the files to copy;
the R and Python bindings and standalone examples are kept separate.

```sh
python tools/sync_core.py
python tools/sync_core.py --check
```

We keep the copied files in version control so that each package can be built
from its source archive without the rest of the repository. We edit the originals
in `c++/`; the build checks identify outdated copies in `bagelR/src/` and
`python/vendor/bagel/`.

The shared code includes `bagel`, `LP-M`, and `LP-P` pruning. The package tests
are in `bagelR/tests/` and `python/tests/`.

## Using Bagel in R

We install the development package from GitHub as follows:

```r
install.packages("remotes")
remotes::install_github("grosed/bagel", subdir = "bagelR")
```

To install changes from a local checkout, we run the following from the
repository root in R:

```r
remotes::install_local("bagelR")
```

Installation from source requires a C++20 toolchain. To build the introductory vignette as well:

```r
install.packages(c("knitr", "rmarkdown"))
remotes::install_github("grosed/bagel", subdir = "bagelR",
                        build_vignettes = TRUE)
vignette("bagelR", package = "bagelR")
```

Vignette building also requires Pandoc. The tutorial source is [Getting started with bagelR](bagelR/vignettes/bagelR.Rmd). It covers mean, slope and seasonal changes, autocorrelated errors, and sequential detection.

```r
library(bagelR)
packageVersion("bagelR")
```

The introduction video is stored outside the R package. [Watch the video online and view its credits](release-assets/bagel-introduction/release-notes.md). The vignette thumbnail opens this GitHub page with a video player.

## Building and checking

We edit R help in the roxygen comments in `bagelR/R/`, then regenerate it
from an R session with `roxygen2` installed:

```r
roxygen2::roxygenise("bagelR", roclets = c("rd", "namespace"))
```

We include the generated `bagelR/man/` files and `NAMESPACE` with the source
changes. From the repository root, we build and check the package:

```sh
python tools/sync_core.py --check
R CMD build bagelR
R CMD check --as-cran bagelR_1.8.2.tar.gz
```

The archive version follows `bagelR/DESCRIPTION`. The vignette requires
Pandoc, and the PDF manual requires LaTeX. Python build and test commands
are in the [Python guide](python/README.md#development-and-releases).
