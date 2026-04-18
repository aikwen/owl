set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
set shell := ["bash", "-c"]

default:
    just --list

install:
    pip install -e .

# 清理工作区：移除旧的构建目录和 egg-info 文件
@clean:
    echo "Cleaning workspace..."
    {{ if os() == "windows" {
        "if (Test-Path dist) { rm -r -force dist };
         if (Test-Path build) { rm -r -force build };
         if (Get-ChildItem *.egg-info) { rm -r -force *.egg-info }"
      } else {
        "rm -rf dist build *.egg-info"
      } }}
    echo "Cleaning success!"

# 打包
package: clean
    pip install --upgrade build twine
    python -m build

# 上传pypi
upload-pypi:
    python -m twine upload --repository pypi dist/*

# 上传testpypi
upload-testpypi:
    python -m twine upload --repository testpypi dist/*