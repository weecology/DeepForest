#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import urllib.request
from typing import Any

from recommonmark.parser import CommonMarkParser
from recommonmark.transform import AutoStructify

from sphinx.highlighting import lexers
from pygments.lexers.python import PythonLexer

lexers["python"]=PythonLexer()

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '..'))
sys.path.insert(0, os.path.abspath('..'))

# Create content for deepforestr.md, skip the badge section
deepforestr_title = """# Using DeepForest from R

An R wrapper for DeepForest is available in the [deepforestr package](https://github.com/weecology/deepforestr).
Commands are very similar with some minor differences due to how the wrapping process
using [reticulate](https://rstudio.github.io/reticulate/) works.

"""
file_obj = open('deepforestr.md', 'w')
readme_url = 'https://raw.githubusercontent.com/weecology/deepforestr/main/README.md'
file_obj.write(deepforestr_title)

with urllib.request.urlopen(readme_url) as response:
    lines = response.readlines()
    badge_section = True
    for line in lines:
        line = line.decode("utf8")
        if "## Installation" in line and badge_section:
            badge_section = False
        if not badge_section:
            file_obj.write(line)
file_obj.close()

# Create copy of CONTRIBUTING.md
contributing_url = "https://raw.githubusercontent.com/weecology/DeepForest/main/CONTRIBUTING.md"
contributing_source = "../CONTRIBUTING.md"

if not os.path.exists(contributing_source):
    with urllib.request.urlopen(contributing_url) as response:
        lines = response.readlines()
        with open(contributing_source, "w") as file1:
            file1.write(lines)

# reading from file1 and writing to file2
with open(contributing_source, "r") as file1:
    with open("CONTRIBUTING.md", "w") as file2:
        file2.write(file1.read())

needs_sphinx = "1.8"

autodoc_default_options = {
    'members': None,
    'show-inheritance': None,
}
autodoc_member_order = 'groupwise'
autoclass_content = 'both'

extensions = [
    'nbsphinx', 'pygments.sphinxext', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary',
    'sphinx.ext.doctest', 'sphinx.ext.githubpages', 'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx', 'sphinx.ext.mathjax', 'sphinx.ext.napoleon',
    'sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx_markdown_tables'
]

nbsphinx_execute = 'never'
nbsphinx_allow_errors = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'DeepForest'
copyright = u"2019, Ben Weinstein"
author = u"Ben Weinstein"
version = u"__version__ = '__version__ = '__version__ = '1.3.3'''"
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
pygments_style = 'sphinx'
todo_include_todos = False

# -- Options for HTML output -------------------------------------------
html_theme = 'furo'
html_static_path = []

# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'deepforestdoc'

# -- Options for LaTeX output ------------------------------------------

latex_elements = {}
latex_documents = [
    (master_doc, 'deepforest.tex', u'DeepForest Documentation', u'Ben Weinstein',
     'manual'),
]

# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
man_pages = [(master_doc, 'deepforest', u'DeepForest Documentation', [author], 1)]

# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'deepforest', u'DeepForest Documentation', author, 'deepforest',
     'One line description of project.', 'Miscellaneous'),
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Temporary workaround to remove multiple build warnings caused by upstream bug
# See https://github.com/zulip/zulip/issues/13263 for details.

# Suppress warnings due to recommonmark config not being cacheable
suppress_warnings = ["config.cache"]


class CustomCommonMarkParser(CommonMarkParser):

    def visit_document(self, node):
        pass


def setup(app: Any) -> None:
    app.add_source_parser(CustomCommonMarkParser)
    app.add_config_value(
        'recommonmark_config',
        {
            'enable_eval_rst': True,
            # Turn off recommonmark features we aren't using.
            'enable_auto_doc_ref': False,
            'auto_toc_tree_section': None,
            'enable_auto_toc_tree': False,
            'enable_math': False,
            'enable_inline_math': False,
            'url_resolver': lambda x: x,
        },
        True)

    # Enable `eval_rst`, and any other features enabled in recommonmark_config.
    # Docs: http://recommonmark.readthedocs.io/en/latest/auto_structify.html
    # (But NB those docs are for master, not latest release.)
    app.add_transform(AutoStructify)

    # overrides for wide tables in RTD theme
    app.add_css_file('theme_overrides.css')  # path relative to _static
