from setuptools import find_packages, setup
from Cython.Build import cythonize
from pathlib import Path
from pathlib import Path
import os

platforms = ["t4", "p4", "l40"]

def UnitName2AlgoType(unit_name:str):
    assert len(unit_name) <= 40,\
        f"invalid unit name, less than 40 characters required, got {len(unit_name)}."
    items = unit_name.split("_")
    assert len(items) > 1,\
        f"invalid unit name {unit_name}, required agi_plt(optinal)_algo_type format."

    algo_type = ""
    if items[1] in platforms:
        algo_type = "_".join(items[2:])
    else:
        algo_type = "_".join(items[1:])

    return algo_type

def GetAlgoInfo():
    '''根据文件夹名字，检查algotype和tasktype，并返回对应的unitype和algotype
    '''
    file_p = Path(__file__)
    unit_name = file_p.parent.name
    algo_type = UnitName2AlgoType(unit_name)
    assert file_p.with_name(algo_type).exists(), f"package dir name:{algo_type} isn't exist！！！"
    
    return unit_name, algo_type


packages = find_packages(exclude=['__pycache__'])
# print(f"packages:{packages}")
def read_file(file):
    assert Path(file).exists(), "requirements.txt isn't exist!!"
    with open(file, "rt") as f:
      return f.read()

requires = [i for i in read_file("requirements.txt").strip().splitlines() if i != '']

unit_name, algo_type = GetAlgoInfo()
enterpoint_py = Path(__file__).with_name(algo_type).joinpath(f"ezviz_backend.py")
assert enterpoint_py.exists(), f"enterpoint py file:{enterpoint_py} not exist"

## 编译规则，控制哪些文件编译成.so,哪些不编译
so_filelist = []
py_filelist = []
c_filelist = []
for pkg in packages:
    pkg = Path(pkg.replace(".",'/'))
    for pyp in pkg.glob("*.py"):
        pyp = str(pyp)
        py_filelist.append(pyp.replace('.py', '').replace('/', '.'))
        # if "ezviz_backend" in pyp:
        #     py_filelist.append(pyp.replace('.py', '').replace('/', '.'))
        # else:
        #     so_filelist.append(pyp)
        #     c_filelist.append(pyp.replace(".py", ".c"))

setup(
    name=f"{unit_name}_backend",
    version="2.6.0",
    description=f"model server - {algo_type} backend",
    author="wangsheng30",
    author_email="wangsheng30@hikvision.com.cn",
    entry_points={
        'MS_Process': [
            f'{unit_name}={algo_type}.ezviz_backend:EzvizModelBackend'
        ]
    },
    py_modules= py_filelist,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: Microsoft',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries'
    ],
    zip_safe=True,
    install_requires= requires,
    ext_modules= cythonize(so_filelist)
)

for c_f in c_filelist:
    os.remove(c_f)
