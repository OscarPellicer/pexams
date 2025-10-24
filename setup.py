from setuptools import setup, find_packages

setup(
    name="pexams",
    version="0.1.0",
    description="A Python library for generating and correcting exams using Playwright and OpenCV.",
    author="AutoTestIA",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
        "pdf2image",
        "pydantic",
        "Markdown",
        "Faker",
        "Pillow",
        "playwright",
        "matplotlib",
        "pandas"
    ],
    extras_require={
        "ocr": ["torch", "torchvision", "transformers", "timm"]
    },
    entry_points={
        'console_scripts': [
            'pexams=pexams.main:main',
        ],
    },
)
