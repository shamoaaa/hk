whl_name:=$(shell basename $(shell pwd))
base:
	rm -f MS_Manager*
	rm -f MS_BE*
	
	## 卸载MS_BE、MS_Manager
	pip uninstall MS_Manager -y
	pip uninstall MS_BE -y

	## 下载包
	wget http://10.222.20.48:20000/AGI/python_debug/MS_BE-1.3.1-cp39-cp39-linux_x86_64.whl
	# wget http://10.222.20.48:20000/AGI/python_deploy/MS_BE-1.3.1-cp39-cp39-linux_x86_64.whl

	wget http://10.222.20.48:20000/AGI/python_debug/MS_Manager-1.4.1-cp39-cp39-linux_x86_64.whl
	# wget http://10.222.20.48:20000/AGI/python_deploy/MS_Manager-1.4.0-cp39-cp39-linux_x86_64.whl

	## 安装MS_BE、MS_Manager
	pip install MS_Manager*.whl
	pip install MS_BE*.whl

	pip uninstall -y urllib3
	pip install urllib3==1.26.16

env:
	export CUDA_VISIBLE_DEVICES=7
	export MS_WEIGHTS=$PWD/weights
	
module:
	pip uninstall ${whl_name}_backend -y
	rm -f ./dist/${whl_name}*.whl
	python setup.py bdist_wheel
	python -m pyc_wheel ./dist/${whl_name}*.whl
	rm -rf build
	rm -rf *.egg-info
	pip install ./dist/${whl_name}*.whl
