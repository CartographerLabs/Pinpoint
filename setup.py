import setuptools

requirements = []
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="Pinpoint",
    version="0.0.4",
    author="James Stevenson",
    author_email="hi@jamesstevenson.me",
    description="A binary classification model designed for identifying extremist content on Twitter.",
    long_description="Python tooling creating an extremism classification model based on the research Understanding the Radical Mind: Identifying Signals to Detect Extremist Content on Twitter by Mariam Nouh, Jason R.C. Nurse, and Michael Goldsmith.",
    long_description_content_type="text/markdown",
    url="https://github.com/user1342/Pinpoint",
    packages=["Pinpoint"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPL-3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)
