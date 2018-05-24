# Releasing

This document explains what do once your [Pull Request](https://www.atlassian.com/git/tutorials/making-a-pull-request/) has been reviewed and all final changes applied. Now you're ready merge your branch into master and release it to the world:

1. Make sure that you have `pandoc` and `pypandoc` installed on Travis: this is needed for readme markdown on PyPI. (See [here](http://pandoc.org/installing.html) and [here](https://pypi.python.org/pypi/pypandoc), respectively, for instructions.)
2. Bump the [version](http://semver.org/) in `__init__.py`, as part of the PR you want to release.
3. Merge your branch into master.
4. Travis shoukd be configured to auto-deploy to PyPI, as long as the tests succeed and the library version is newer than any released.
