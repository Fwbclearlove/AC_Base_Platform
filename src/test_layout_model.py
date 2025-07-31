# test_environment.py
import sys
import subprocess


def test_import(module_name, import_name=None):
    """测试模块导入"""
    if import_name is None:
        import_name = module_name

    try:
        __import__(import_name)
        print(f"{module_name} 导入成功")
        return True
    except ImportError as e:
        print(f"{module_name} 导入失败: {e}")
        return False


def test_ghostscript():
    """测试Ghostscript"""
    commands = ['gs', 'gswin64c']
    for cmd in commands:
        try:
            result = subprocess.run([cmd, '--version'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"Ghostscript ({cmd}) 可用")
                return True
        except Exception:
            continue
    print("Ghostscript 不可用")
    return False


def main():
    print("=== PDF处理环境测试 ===")
    print(f"Python版本: {sys.version}")
    print(f"当前环境: {sys.prefix}")
    print()

    # 测试核心库
    modules = [
        ("PyMuPDF", "fitz"),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("tqdm", "tqdm"),
        ("camelot", "camelot"),
        ("pdfplumber", "pdfplumber"),
        ("langchain", "langchain"),
        ("sentence-transformers", "sentence_transformers"),
        ("faiss", "faiss"),
        ("zhipuai", "zhipuai")
    ]

    results = []
    for display_name, import_name in modules:
        results.append(test_import(display_name, import_name))

    print()
    # 测试Ghostscript
    gs_result = test_ghostscript()
    results.append(gs_result)

    print()
    print("=== 总结 ===")
    success_count = sum(results)
    total_count = len(results)

    if success_count == total_count:
        print(f"所有依赖配置成功！({success_count}/{total_count})")
        print("可以开始使用混合提取方案了！")
    else:
        print(f"还有 {total_count - success_count} 个依赖需要解决")


if __name__ == "__main__":
    main()