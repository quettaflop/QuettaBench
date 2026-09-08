import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "QuettaBench"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_member_order = "bysource"

# index.rst links api/src directly; the apidoc wrapper page is unused.
exclude_patterns = ["api/modules.rst"]

html_theme = "furo"
