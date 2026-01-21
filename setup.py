"""Setup configuration for maya_grass_gen package."""

import re
from pathlib import Path

from setuptools import find_namespace_packages, setup

REQUIREMENTS_FILE = "requirements.in"


def get_requirements(requirements_file):
    """Loads the requirements from a given file."""
    requirements_content = Path(requirements_file).read_text()
    # Substitutes local overrides with package names
    requirements_content = re.sub(
        r".*?file:.*#egg=([\d\w\.]+).*?\s",
        r"\1\n",
        requirements_content,
        flags=re.MULTILINE,
    )
    # Substitutes all comments with an empty string
    requirements = re.sub(
        r"#.*\n?", "\n", requirements_content, flags=re.MULTILINE
    ).splitlines()
    # Filters any empty strings
    return list(filter(bool, map(str.strip, requirements)))


setup(
    name="maya_grass_gen",
    version="0.1.0",
    description="Maya grass generation plugin using flow fields and MASH instancing",
    author="Jason Shipp",
    author_email="bit.shonas@gmail.com",
    python_requires="~=3.12",
    include_package_data=True,
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    install_requires=get_requirements(REQUIREMENTS_FILE),
    zip_safe=False,
)
