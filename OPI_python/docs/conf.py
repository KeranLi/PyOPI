"""
Sphinx configuration for OPI documentation.
"""

import os
import sys

# Add the parent directory to the path so Sphinx can find the opi package
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'OPI (Orographic Precipitation and Isotopes)'
copyright = '2026, Mark Brandon, Yale University (MATLAB); AI Assistant (Python)'
author = 'Mark Brandon, Yale University (MATLAB); AI Assistant (Python)'

# The full version, including alpha/beta/rc tags
version = '2.0.0'
release = '2.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here
extensions = [
    'sphinx.ext.autodoc',      # Automatically document modules
    'sphinx.ext.napoleon',     # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',     # Add links to source code
    'sphinx.ext.mathjax',      # Render math equations
    'sphinx.ext.intersphinx',  # Link to other project's documentation
    'sphinx.ext.autosummary',  # Generate summary tables
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon settings for docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# Add any paths that contain templates here
templates_path = ['_templates']

# The suffix of source filenames
source_suffix = '.rst'

# The master toctree document
master_doc = 'index'

# List of patterns to exclude
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Add any paths that contain custom static files
html_static_path = ['_static']

# Custom CSS files
html_css_files = [
    'custom.css',
]

# The name of an image file to use as a favicon
# html_favicon = '_static/favicon.ico'

# The name of an image file to place at the top of the sidebar
# html_logo = '_static/logo.png'

# Output file base name for HTML help builder
htmlhelp_basename = 'OPIdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper')
    'papersize': 'letterpaper',
    
    # The font size ('10pt', '11pt' or '12pt')
    'pointsize': '10pt',
    
    # Additional stuff for the LaTeX preamble
    'preamble': '',
    
    # Latex figure (float) alignment
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files
latex_documents = [
    (master_doc, 'OPI.tex', 'OPI Documentation',
     'Mark Brandon, Yale University', 'manual'),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'opi', 'OPI Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (master_doc, 'OPI', 'OPI Documentation',
     author, 'OPI', 'Orographic Precipitation and Isotopes Model',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False
