from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import os

# ================= 配置区域 =================
package_root = "src/radar_pig_pkg"
# ===========================================

# 1. 定义编译选项
compiler_directives = {
    'language_level': 3,
    'embedsignature': True,
}

# 2. 收集所有需要编译的 .py 文件
extensions = []

if not os.path.exists(package_root):
    print(f"错误：找不到目录 {package_root}")
else:
    for root, dirs, files in os.walk(package_root):
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'tests']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # 计算模块名
                rel_path = os.path.relpath(file_path)
                module_name = rel_path[:-3].replace(os.sep, '.').replace('/', '.').replace('\\', '.')
                if module_name.startswith("src."):
                    module_name = module_name[4:]

                print(f"正在添加模块: {module_name} <- {file_path}")

                ext = Extension(
                    module_name,
                    [file_path],
                    extra_compile_args=[],
                )
                extensions.append(ext)

# 3. 使用 Cython 处理
ext_modules = cythonize(
    extensions,
    compiler_directives=compiler_directives,
    annotate=False,
    force=True
)

# 4. Setup 配置
setup(
    name="radar_pig_pkg",
    version="0.1.0",
    description="radar pig package",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    # 注意：这里不要加 exclude_package_data，让它正常打包
    ext_modules=ext_modules,
    install_requires=[
        "pandas", "numpy", "dtaidistance", "h5py", 
        "matplotlib", "oss2", "scikit_learn", "scipy", "xgboost"
    ],
    zip_safe=False,
)
