# download_layout_model.py (已验证和修正)

import layoutparser as lp
import os

print("=" * 50)
print("开始下载 LayoutParser 的 PaddleDetection 版面分析模型...")
print("这个过程将使用layoutparser的官方API，请耐心等待。")
print("=" * 50)

try:
    # 这行代码是关键。它会：
    # 1. 自动从官方源下载模型配置文件和权重文件。
    # 2. 将它们缓存到本地的一个默认目录中。
    # 3. 返回一个加载了模型的对象。
    # 我们选择一个在PubLayNet上训练的、强大的ppyolov2模型。
    model = lp.PaddleDetectionLayoutModel(
        config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        enforce_cpu=True,  # 强制在CPU上运行，避免GPU依赖问题
    )

    print("\n[成功] 模型已成功下载并加载！")

    # 找到缓存路径并打印出来，方便我们去复制文件
    # layoutparser会将模型下载到 user_home/.layoutparser/models/ 目录下
    model_path = model.config.config_path.parent
    print(f"\n[信息] 模型文件已缓存到以下目录:")
    print(f"      {model_path}")

    print("\n[下一步] 请将以上目录中的所有文件和文件夹，复制到您项目电脑的")
    print(r"      C:\Users\YourName\.layoutparser\models\ 目录下。")
    print("=" * 50)

except Exception as e:
    print(f"\n[失败] 下载或加载模型时出错: {e}")
    print("\n请检查您的网络连接，或尝试更换网络环境（如手机热点）。")
    print("如果反复失败，请检查paddlepaddle和layoutparser是否安装正确。")