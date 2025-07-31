# src/test_api.py
import os
from zhipuai import ZhipuAI, ZhipuAIError

# 使用你的API Key
API_KEY = "a5528da8417645d8a4dfb71a9d30f140.7i6b5zfJDRYLgE9C"
CLIENT = ZhipuAI(api_key=API_KEY)

def run_test():
    """
    执行一个最简单的API调用测试。
    """
    print("--- 开始进行最小化的API连接测试 ---")
    try:
        response = CLIENT.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "user", "content": "你好，请问你是谁？"}
            ],
            temperature=0.9,
            max_tokens=25,
        )
        print("\n[成功] API调用成功！")
        print("API返回的回答是:")
        print(response.choices[0].message.content)

    except ZhipuAIError as e:
        print("\n[失败] 捕获到ZhipuAI的特定错误！")
        print(f"错误类型: {type(e)}")
        print(f"错误代码: {e.code}")
        print(f"错误信息: {e.message}")

    except Exception as e:
        print("\n[失败] 捕获到未知的全局异常！")
        print(f"异常类型: {type(e)}")
        print(f"异常详情: {repr(e)}")

    print("\n--- 测试结束 ---")

if __name__ == "__main__":
    run_test()