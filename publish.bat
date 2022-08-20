conda activate tf2
python setup.py sdist bdist_wheel
twine upload dist/* -u $PYPI_USER -p $PYPI_PASSWORD
rd /q /s dist build bert4tf2.egg-info
conda deactivate