#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import urllib.request
from typing import Any, Dict

from recommonmark.parser import CommonMarkParser
from recommonmark.transform import AutoStructify
from sphinx.highlighting import lexers
from pygments.lexers.python import PythonLexer
from deepforest._version import __version__

# ----------------------------------
# Syntax Highlighting Configuration
# ----------------------------------
lexers["python"] = PythonLexer()

# ----------------------------------
# Path Setup
# ----------------------------------
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, ".."))
sys.path.insert(0, os.path.abspath(".."))

# ----------------------------------
# Generate deepforestr.md File
# ----------------------------------
file_path = "user_guide/deepforestr.md"
readme_url = "https://raw.githubusercontent.com/weecology/deepforestr/main/README.md"

deepforestr_title = """# Using DeepForest from R

An R wrapper for DeepForest is available in the [deepforestr package](https://github.com/weecology/deepforestr).
Commands are very similar with some minor differences due to how the wrapping process
using [reticulate](https://rstudio.github.io/reticulate/) works.
"""

with open(file_path, "w") as file_obj:
    file_obj.write(deepforestr_title)
    try:
        with urllib.request.urlopen(readme_url) as response:
            lines = response.readlines()
            badge_section = True
            for line in lines:
                line = line.decode("utf-8")
                if "## Installation" in line:
                    badge_section = False
                if not badge_section:
                    file_obj.write(line)
    except Exception:
        print("Could not retrieve the deepforestr README, skipping")
        json_url = "../version_switcher.json"

# ----------------------------------
# Sphinx Configuration
# ----------------------------------
needs_sphinx = "1.8"

autodoc_default_options = {"members": None, "show-inheritance": None}
autodoc_member_order = "groupwise"
autoclass_content = "both"

extensions = [
    "sphinx_design",
    "nbsphinx",
    "pygments.sphinxext",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_markdown_tables",
]

nbsphinx_execute = "never"
nbsphinx_allow_errors = True

templates_path = ["_templates"]
master_doc = "index"

# ----------------------------------
# Project Information
# ----------------------------------
project = "DeepForest"
copyright = "2019, Ben Weinstein"
author = "Ben Weinstein"
version = release = str(__version__.replace("-dev0", ""))

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
pygments_style = "sphinx"
todo_include_todos = False

# ----------------------------------
# HTML Output Options
# ----------------------------------
html_theme = "pydata_sphinx_theme"

# Version Switcher Logic
if ".dev" in version:
    switcher_version = "dev"
elif "rc" in version:
    switcher_version = version.split("rc", maxsplit=1)[0] + " (rc)"
else:
    switcher_version = ".".join(version.split(".")[:2])

html_static_path = ["_static"]
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_align": "left",
    "show_version_warning_banner": True,
    "header_links_before_dropdown": 5,
    "secondary_sidebar_items": [
        "page-toc",
        "searchbox",
        "edit-this-page",
        "sourcelink",
    ],
    "github_url": "https://github.com/weecology/DeepForest",
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
}

html_sidebars: Dict[str, Any] = {
    "index": [],
    "**": ["sidebar-nav-bs.html"],
}

html_css_files = [
    "css/getting_started.css",
    "css/pandas.css",
]

# ----------------------------------
# Documentation Output Configuration
# ----------------------------------
htmlhelp_basename = "deepforestdoc"
latex_documents = [
    (
        master_doc,
        "deepforest.tex",
        "DeepForest Documentation",
        "Ben Weinstein",
        "manual",
    )
]

man_pages = [(master_doc, "deepforest", "DeepForest Documentation", [author], 1)]

txinfo_documents = [
    (
        master_doc,
        "deepforest",
        "DeepForest Documentation",
        author,
        "deepforest",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# ----------------------------------
# Source Suffix Configuration
# ----------------------------------
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# Suppress Warnings
suppress_warnings = ["config.cache", "toc.not_readable"]

# ----------------------------------
# Custom CommonMark Parser
# ----------------------------------
class CustomCommonMarkParser(CommonMarkParser):
    def visit_document(self, node):
        pass

def setup(app: Any) -> None:
    app.add_source_parser(CustomCommonMarkParser)
    app.add_config_value(
        "recommonmark_config",
        {
            "enable_eval_rst": True,
            "enable_auto_doc_ref": False,
            "auto_toc_tree_section": None,
            "enable_auto_toc_tree": False,
            "enable_math": False,
            "enable_inline_math": False,
            "url_resolver": lambda x: x,
        },
        True,
    )
    app.add_transform(AutoStructify)
    app.add_css_file("theme_overrides.css")
