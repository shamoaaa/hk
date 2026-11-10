import os
import shutil
import zipfile
import glob

# ================= 配置区域 =================
# 你的 wheel 包名称（不需要写版本号和平台后缀，脚本会自动匹配）
PACKAGE_NAME = "radar_pig_pkg"
# ===========================================

def clean_wheel():
    print("1. 开始清理旧的构建文件...")
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    # 清理源码目录下的 .so 文件（可选，防止旧版本干扰）
    # for root, dirs, files in os.walk("src"):
    #     for f in files:
    #         if f.endswith(".so") or f.endswith(".pyd"):
    #             os.remove(os.path.join(root, f))

    print("2. 正在构建 Wheel (这可能需要几分钟)...")
    os.system("python setup.py bdist_wheel")

    print("3. 寻找生成的 .whl 文件...")
    wheel_files = glob.glob(f"dist/{PACKAGE_NAME}-*.whl")
    if not wheel_files:
        print("错误：未找到生成的 wheel 文件！")
        return

    wheel_path = wheel_files[0]
    print(f"找到 Wheel: {wheel_path}")

    # 创建一个临时目录
    temp_dir = "temp_wheel_extract"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    print("4. 解压 Wheel 文件...")
    with zipfile.ZipFile(wheel_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    print("5. 扫描并删除 .py 源文件...")
    deleted_count = 0
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # 删除 .py 文件
                os.remove(file_path)
                print(f"   - 删除源码: {file_path}")
                deleted_count += 1
    
    if deleted_count == 0:
        print("   (警告：没有找到任何 .py 文件，可能已经被编译或路径不对)")

    print("6. 重新打包成 Wheel...")
    new_wheel_path = wheel_path  # 覆盖原文件
    with zipfile.ZipFile(new_wheel_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                abs_file = os.path.join(root, file)
                # 计算在 zip 中的相对路径
                rel_file = os.path.relpath(abs_file, temp_dir)
                zipf.write(abs_file, rel_file)

    print("7. 清理临时文件...")
    shutil.rmtree(temp_dir)

    print(f"\n✅ 成功！清洗后的 Wheel 已生成: {new_wheel_path}")
    print(f"   共删除了 {deleted_count} 个源码文件。")

if __name__ == "__main__":
    clean_wheel()
