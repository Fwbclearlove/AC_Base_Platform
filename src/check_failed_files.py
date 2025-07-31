# check_failed_files.py
import json
from pathlib import Path


def check_processing_results():
    """检查文本提取结果，列出失败的文件"""

    metadata_path = Path(r"D:\AC_Base_Platform\data\processed\metadata.json")

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("找不到metadata.json文件")
        return

    # 统计各种状态
    status_count = {}
    failed_files = []
    successful_files = []

    for file_id, info in metadata.items():
        status = info.get("status", "unknown")
        status_count[status] = status_count.get(status, 0) + 1

        if status == "text_extracted":
            successful_files.append({
                'file': file_id,
                'length': info.get('text_length', 0)
            })
        else:
            failed_files.append({
                'file': file_id,
                'status': status,
                'error': info.get('error', 'Unknown error')
            })

    # 打印统计结果
    print("=" * 60)
    print("文本提取结果统计")
    print("=" * 60)

    print(f"总文件数: {len(metadata)}")
    print("\n各状态文件数量:")
    for status, count in status_count.items():
        print(f"  {status}: {count}")

    print(f"\n成功率: {len(successful_files)}/{len(metadata)} ({len(successful_files) / len(metadata) * 100:.1f}%)")

    # 显示失败文件详情
    if failed_files:
        print("\n" + "=" * 60)
        print("处理失败的文件:")
        print("=" * 60)
        for i, file_info in enumerate(failed_files, 1):
            print(f"{i:2d}. {file_info['file']}")
            print(f"    状态: {file_info['status']}")
            print(f"    错误: {file_info['error']}")
            print()

    # 显示成功文件的一些统计
    if successful_files:
        print("\n" + "=" * 60)
        print("成功处理的文件统计:")
        print("=" * 60)

        lengths = [f['length'] for f in successful_files]
        avg_length = sum(lengths) / len(lengths)
        max_length = max(lengths)
        min_length = min(lengths)

        print(f"平均文本长度: {avg_length:.0f} 字符")
        print(f"最长文本: {max_length} 字符")
        print(f"最短文本: {min_length} 字符")

        # 找出最长和最短的文件
        longest_file = max(successful_files, key=lambda x: x['length'])
        shortest_file = min(successful_files, key=lambda x: x['length'])

        print(f"\n最长文件: {longest_file['file']} ({longest_file['length']} 字符)")
        print(f"最短文件: {shortest_file['file']} ({shortest_file['length']} 字符)")


if __name__ == "__main__":
    check_processing_results()