from setuptools import setup, find_packages

setup(
    name='bert4tf2',
    version='0.0.1',
    description='A very simple and easy to learn BERT library for tensorflow 2.x',
    license='Apache License 2.0',
    url='https://github.com/Luffffffy/bert4tf',
    author='Luffy',
    author_email='571036709@qq.com',
    install_requires=['tensorflow<=2.9.0'],
    packages=find_packages()
)