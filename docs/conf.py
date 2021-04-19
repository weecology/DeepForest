#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from typing import Any

import recommonmark
from recommonmark.parser import CommonMarkParser
from recommonmark.parser import CommonMarkParser
from recommonmark.parser import CommonMarkParser
from recommonmark.transform import AutoStructify

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '..'))
sys.path.insert(0, os.path.abspath('..'))

needs_sphinx = "1.8"

autodoc_default_options = {
    'members': None,
    'show-inheritance': None,
}
autodoc_member_order = 'groupwise'
autoclass_content = 'both'

extensions = [
    'pygments.sphinxext',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.githubpages',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_markdown_tables',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'DeepForest'
copyright = u"2019, Ben Weinstein"
author = u"Ben Weinstein"
version = u"__version__ = '0.1.31'"
language = None
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
pygments_style = 'sphinx'
todo_include_todos = False

# -- Options for HTML output -------------------------------------------

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme

    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()] + ["../.."]

html_static_path = []

# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'deepforestdoc'

# -- Options for LaTeX output ------------------------------------------

latex_elements = {}
latex_documents = [
    (master_doc, 'deepforest.tex', u'DeepForest Documentation',
     u'Ben Weinstein', 'manual'),
]

# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
man_pages = [(master_doc, 'deepforest', u'DeepForest Documentation', [author],
              1)]

# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'deepforest', u'DeepForest Documentation', author,
     'deepforest', 'One line description of project.', 'Miscellaneous'),
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Temporary workaround to remove multiple build warnings caused by upstream bug.
# See https://github.com/zulip/zulip/issues/13263 for details.


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
    app.add_stylesheet('theme_overrides.css')  # path relative to _static
