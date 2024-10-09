from setuptools import setup, find_packages

setup(
    name="hworldmodel",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here, e.g., 'numpy', 'pandas',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A brief description of your project",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/lacykaltgr/hworldmodel",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)