# src/test_zhipu_api.py

import os
from zhipuai import ZhipuAI, ZhipuAIError

# --- 配置 ---
# 请将这里替换为您在04b脚本中使用的同一个API Key
# 警告：API Key已硬编码。仅用于临时测试。
API_KEY = "838cc9e6876a4fea971b3728af105b56.1KDgfLzNHnfllnhb"

# 要测试的模型
MODEL_NAME = "glm-4"


def run_api_test():
    """
    执行一个最小化的、完整的智谱AI API调用测试。
    """
    print("=" * 50)
    print("--- 开始进行智谱AI (ZhipuAI) API连接测试 ---")
    print(f"--- 使用模型: {MODEL_NAME} ---")
    print("=" * 50)

    # 1. 检查API Key是否已设置
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        print("\n[失败] 错误：API Key未在脚本中设置。")
        print("请在脚本中填入您的智谱AI API Key。")
        return

    try:
        # 2. 初始化客户端
        print("\n[步骤1/3] 正在初始化ZhipuAI客户端...")
        client = ZhipuAI(api_key=API_KEY)
        print("客户端初始化成功！")

        # 3. 构造一个简单的对话请求
        test_messages = [
            {"role": "user", "content": "你好，请用一句话介绍一下你自己。"}
        ]

        print("\n[步骤2/3] 正在向API发送一个简单的测试请求...")
        print(f"发送内容: {test_messages[0]['content']}")

        # 4. 发送请求并接收响应
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=test_messages,
            temperature=0.7,  # 稍微增加一点随机性
            max_tokens=50,  # 限制最大输出长度
        )

        print("API响应接收成功！")

        # 5. 解析并打印结果
        print("\n[步骤3/3] 解析API响应...")

        if response and response.choices:
            reply_content = response.choices[0].message.content
            usage_info = response.usage

            print("\n" + "=" * 50)
            print("测试成功！API工作正常！")
            print("=" * 50)
            print("\n模型返回的回答是:")
            print(f"> {reply_content}")

            print("\n本次调用消耗的Token信息:")
            print(f"  - 提示 (Prompt) tokens: {usage_info.prompt_tokens}")
            print(f"  - 回答 (Completion) tokens: {usage_info.completion_tokens}")
            print(f"  - 总计 (Total) tokens: {usage_info.total_tokens}")

        else:
            print("\n[失败] 错误：API返回了空的或无效的响应。")
            print(f"原始响应: {response}")

    except ZhipuAIError as e:
        print("\n" + "=" * 50)
        print("测试失败：捕获到ZhipuAI的特定错误！")
        print("=" * 50)
        print(f"\n错误类型: {type(e)}")
        print(f"错误代码: {e.code}")
        print(f"错误信息: {e.message}")
        print("\n[可能的原因]")
        print(" - API Key无效或已过期。")
        print(" - 账户余额不足。")
        print(" - 请求的模型不存在或无权限访问。")

    except Exception as e:
        print("\n" + "=" * 50)
        print("测试失败：捕获到未知的全局异常！")
        print("=" * 50)
        print(f"\n异常类型: {type(e)}")
        print(f"异常详情: {repr(e)}")
        print("\n[可能的原因]")
        print(" - 网络连接问题（防火墙、代理、无法连接到服务器）。")
        print(" - `zhipuai`库安装不正确或版本冲突。")

    print("\n--- 测试结束 ---")


if __name__ == "__main__":
    run_api_test()