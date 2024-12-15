from setuptools import setup, find_packages

setup(
    name="tdsproject2",
    version="1.0.0",
    author="Ashutosh Maurya",
    author_email="your_email@example.com",
    description="Automated Data Analysis and Visualization Script using OpenAI APIs.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tdsproject2",  # Replace with your actual repo URL
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "openai>=1.0.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.7'
)
