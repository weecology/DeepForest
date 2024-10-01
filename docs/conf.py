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

# Set the lexer for Python syntax highlighting
lexers["python"] = PythonLexer()

# Set paths
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '..'))
sys.path.insert(0, os.path.abspath('..'))

# Create content for deepforestr.md, skipping the badge section
deepforestr_title = """# Using DeepForest from R

An R wrapper for DeepForest is available in the [deepforestr package](https://github.com/weecology/deepforestr).
Commands are very similar with some minor differences due to how the wrapping process
using [reticulate](https://rstudio.github.io/reticulate/) works.

"""

file_path = 'user_guide/deepforestr.md'
readme_url = 'https://raw.githubusercontent.com/weecology/deepforestr/main/README.md'

with open(file_path, 'w') as file_obj:
    file_obj.write(deepforestr_title)

    with urllib.request.urlopen(readme_url) as response:
        lines = response.readlines()
        badge_section = True
        for line in lines:
            line = line.decode("utf-8")
            if "## Installation" in line:
                badge_section = False
            if not badge_section:
                file_obj.write(line)

# Sphinx configuration
needs_sphinx = "1.8"
autodoc_default_options = {'members': None, 'show-inheritance': None}
autodoc_member_order = 'groupwise'
autoclass_content = 'both'

extensions = [
    "sphinx_design", 'nbsphinx', 'pygments.sphinxext', 'sphinx.ext.autodoc',
    'sphinx.ext.autosummary', 'sphinx.ext.doctest', 'sphinx.ext.githubpages',
    'sphinx.ext.inheritance_diagram', 'sphinx.ext.intersphinx', 'sphinx.ext.mathjax',
    'sphinx.ext.napoleon', 'sphinx.ext.todo', 'sphinx.ext.viewcode',
    'sphinx_markdown_tables'
]

nbsphinx_execute = 'never'
nbsphinx_allow_errors = True

templates_path = ['_templates']
master_doc = 'index'

# Project information
project = 'DeepForest'
copyright = "2019, Ben Weinstein"
author = "Ben Weinstein"
version = release = str(__version__.replace("-dev0", ""))

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
pygments_style = 'sphinx'
todo_include_todos = False

# HTML output options
html_theme = 'pydata_sphinx_theme'
json_url = 'https://raw.githubusercontent.com/weecology/DeepForest/refs/heads/main/version_switcher.json'

if ".dev" in version:
    switcher_version = "dev"
elif "rc" in version:
    switcher_version = version.split("rc", maxsplit=1)[0] + " (rc)"
else:
    switcher_version = ".".join(version.split(".")[:2])

html_static_path = []
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_align": "left",
    "show_version_warning_banner": True,
    "header_links_before_dropdown": 5,
    "secondary_sidebar_items": ["page-toc", "searchbox", "edit-this-page", "sourcelink"],
    "github_url": "https://github.com/weecology/DeepForest",
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "switcher": {
        "json_url": f"{json_url}",
        "version_match": switcher_version,
    },
}

html_sidebars: Dict[str, Any] = {
    "index": [],
    "**": ["sidebar-nav-bs.html"],
}
html_static_path = ["_static"]
html_css_files = [
    "css/getting_started.css",
    "css/pandas.css",
]

# Output file base names for builders
htmlhelp_basename = 'deepforestdoc'
latex_documents = [(master_doc, 'deepforest.tex', 'DeepForest Documentation', 'Ben Weinstein', 'manual')]
man_pages = [(master_doc, 'deepforest', 'DeepForest Documentation', [author], 1)]
texinfo_documents = [
    (master_doc, 'deepforest', 'DeepForest Documentation', author, 'deepforest', 'One line description of project.', 'Miscellaneous'),
]

# Source suffix configuration
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}

# Suppress warnings due to recommonmark config not being cacheable
suppress_warnings = ["config.cache", "toc.not_readable"]


# Custom CommonMark parser
class CustomCommonMarkParser(CommonMarkParser):
    def visit_document(self, node):
        pass


def setup(app: Any) -> None:
    app.add_source_parser(CustomCommonMarkParser)
    app.add_config_value(
        'recommonmark_config',
        {
            'enable_eval_rst': True,
            'enable_auto_doc_ref': False,
            'auto_toc_tree_section': None,
            'enable_auto_toc_tree': False,
            'enable_math': False,
            'enable_inline_math': False,
            'url_resolver': lambda x: x,
        },
        True
    )
    app.add_transform(AutoStructify)
    app.add_css_file('theme_overrides.css')
