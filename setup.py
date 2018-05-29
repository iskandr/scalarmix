# Copyright (c) 2018. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function, absolute_import
import os
import re

from setuptools import setup, find_packages

project_name = "scalarmix"
description = "Scalar bimodal Gaussian mixture models"
author = "Alex Rubinsteyn"
author_email = "alex@openvax.org"

project_name_no_dashes = project_name.replace("-", "_")
current_directory = os.path.dirname(__file__)
readme_filename = 'README.md'
readme_path = os.path.join(current_directory, readme_filename)

try:
    with open(readme_path, 'r') as f:
        readme_markdown = f.read()
except IOError as e:
    print(e)
    print("Failed to open %s" % readme_path)
    readme_markdown = ""

try:
    import pypandoc
    readme_restructured = pypandoc.convert(readme_markdown, to='rst', format='md')
except ImportError as e:
    readme_restructured = readme_markdown
    print(e)
    print("Failed to convert %s to reStructuredText" % readme_filename)
    pass

def parse_field(name):

    with open('%s/__init__.py' % project_name_no_dashes, 'r') as f:
        value = re.search(
            r'^%s\s*=\s*[\'"]([^\'"]*)[\'"]' % name,
            f.read(),
            re.MULTILINE).group(1)
    if not value:
        raise RuntimeError("Cannot find '%s' in %s's __init__.py" % (name, project_name))
    return value

version = parse_field("__version__")

if not version:
    raise RuntimeError('Cannot find version information')

if __name__ == '__main__':
    setup(
        name=project_name_no_dashes,
        version=version,
        description=description,
        author=author,
        author_email=author_email,
        url="https://github.com/openvax/%s" % project_name,
        license="http://www.apache.org/licenses/LICENSE-2.0.html",
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
        ],
        install_requires=[
            "six>=1.9.0",
            "numpy",
        ],
        long_description=readme_markdown,
        long_description_content_type="text/markdown",
        packages=find_packages(),
    )
