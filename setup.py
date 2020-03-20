import setuptools

with open("README.md", "r") as file_readme:
    long_description = file_readme.read()

setuptools.setup(
    name="fcn_microct", # Replace with your own username
    version="0.0.1",
    author="Alexandre de Siqueira, Daniela Ushizima, Stéfan van der Walt",
    author_email="alex.desiqueira@igdore.org, dushizima@lbl.gov, stefanv@berkeley.edu",
    description="Fully Convolutional Networks for micro-CT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
