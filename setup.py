from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        # ... other parameters ...
        packages=find_packages(include=['buildingsysid', 'buildingsysid.*', 'buildingsysid.calculate.*']),
        include_package_data=True,
    )

# from setuptools import setup, find_packages

# setup(
#     name="buildingsysid",
#     version="0.1.1",
#     description="System identification tools for building models",
#     long_description=open("README.md").read(),
#     long_description_content_type="text/markdown",
#     author="Michael Dahl Knudsen",
#     author_email="mdk@cae.au.dk",
#     url="https://github.com/michael666nan/buildingsysid",
#     packages=find_packages(),
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: MIT License",
#         "Operating System :: OS Independent",
#     ],
#     python_requires=">=3.7",
#     install_requires=[
#         "numpy",
#         "pandas",
#         "matplotlib",
#         "scipy",
#         "statsmodels",
#     ],
# )