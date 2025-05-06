from setuptools import setup, find_packages

setup(
    name="semaclust",
    version="0.1.0",
    author="Mert Cobanov",
    description="Semantic text clustering using sentence embeddings and agglomerative clustering.",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.0",
        "sentence-transformers>=2.2",
    ],
    python_requires=">=3.7",
)
