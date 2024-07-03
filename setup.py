import setuptools


with open('README.md', 'r') as f:
    long_description = f.read()
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f]

setuptools.setup(
    name='nc-toolbox',
    version='0.0.1',
    author='Felix Hauser',
    author_email='felix.hauser@kit.edu',
    description='A Python library to calculate Neural Collapse (NC) related metrics.',
    long_description_content_type='text/markdown',
    long_description=long_description,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)