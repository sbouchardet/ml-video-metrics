import setuptools


setuptools.setup(
    name="ml-video-metrics",  # Replace with your own username
    version="0.0.1",
    author="Susana Bouchardet",
    author_email="susana.bouchardet@gmail.com",
    description="CLI to generate the usual metrics to ML algorithms applied to videos",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "ml-video-metrics = ml_video_metrics.main:main",
        ],
    },
)
