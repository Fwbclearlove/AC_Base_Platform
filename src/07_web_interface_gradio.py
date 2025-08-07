# src/07_web_interface_gradio.py
import gradio as gr
import sys
import os
import importlib

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 修正导入方式
try:
    # 动态导入06_qa_system
    qa_module = importlib.import_module('06_qa_system')
    SimpleQAInterface = qa_module.SimpleQAInterface
except ImportError as e:
    print(f"导入QA系统失败: {e}")
    # 备用方案：直接执行文件内容
    exec(open('06_qa_system.py').read())

import json
from datetime import datetime

# 初始化QA系统
qa_interface = SimpleQAInterface()

# 存储对话历史
conversation_history = []


def answer_question(
        question: str,
        user_level: str,
        search_type: str,
        top_k: int,
        history: list
):
    """处理用户问题并返回答案"""

    if not question:
        return history, "", "请输入问题"

    try:
        # 调用QA系统
        result = qa_interface.ask(
            question=question,
            user_level=user_level.lower(),
            search_type=search_type,
            top_k=top_k
        )

        # 格式化输出
        answer = result['answer']

        # 添加元信息
        meta_info = f"\n\n---\n置信度: {result['confidence']} | 检索质量: {result['search_quality']}"

        # 添加引用
        if result['citations']:
            meta_info += "\n引用来源: "
            for cite in result['citations']:
                meta_info += f"\n- {cite['source_id']}"

        full_answer = answer + meta_info

        # 更新历史
        history.append([question, full_answer])

        # 生成建议问题
        suggestions = "\n".join([f"• {s}" for s in result['suggestions']]) if result['suggestions'] else "暂无建议"

        return history, "", suggestions

    except Exception as e:
        import traceback
        error_msg = f"错误: {str(e)}\n{traceback.format_exc()}"
        history.append([question, error_msg])
        return history, "", "处理出错"


def clear_conversation():
    """清空对话"""
    qa_interface.start_new_conversation()
    return [], "", "对话已清空"


def export_conversation(history):
    """导出对话历史"""
    if not history:
        return "没有对话记录"

    export_data = {
        "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "conversation": [
            {"question": q, "answer": a} for q, a in history
        ]
    }

    filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    return f"对话已导出到: {filename}"


# 创建Gradio界面
with gr.Blocks(title="学术知识问答系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎓 学术知识问答系统

    基于RAG技术的智能学术助手，帮助您快速理解学术知识和研究内容。
    """)

    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                label="对话历史",
                height=500,
                elem_id="chatbot"
            )

            with gr.Row():
                question_input = gr.Textbox(
                    label="请输入您的问题",
                    placeholder="例如：李睿凡的研究方向是什么？",
                    lines=2,
                    scale=4
                )
                submit_btn = gr.Button("提问", variant="primary", scale=1)

        with gr.Column(scale=3):
            gr.Markdown("### 设置")

            user_level = gr.Radio(
                ["Student", "Researcher", "General"],
                label="用户级别",
                value="General",
                info="选择适合您的回答深度"
            )

            search_type = gr.Dropdown(
                ["hybrid", "academic_profile", "semantic", "keyword"],
                label="搜索类型",
                value="hybrid",
                info="选择搜索策略"
            )

            top_k = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="检索数量",
                info="检索相关文档的数量"
            )

            gr.Markdown("### 推荐问题")
            suggestions_box = gr.Textbox(
                label="您可能还想问",
                lines=3,
                interactive=False
            )

            with gr.Row():
                clear_btn = gr.Button("清空对话", size="sm")
                export_btn = gr.Button("导出对话", size="sm")

            export_status = gr.Textbox(label="导出状态", interactive=False)

    # 绑定事件
    submit_btn.click(
        answer_question,
        inputs=[question_input, user_level, search_type, top_k, chatbot],
        outputs=[chatbot, question_input, suggestions_box]
    )

    question_input.submit(
        answer_question,
        inputs=[question_input, user_level, search_type, top_k, chatbot],
        outputs=[chatbot, question_input, suggestions_box]
    )

    clear_btn.click(
        clear_conversation,
        outputs=[chatbot, question_input, export_status]
    )

    export_btn.click(
        export_conversation,
        inputs=[chatbot],
        outputs=[export_status]
    )

    # 示例问题
    gr.Examples(
        examples=[
            ["李睿凡的主要研究方向是什么？"],
            ["BERT模型在NLP中有哪些应用？"],
            ["团队在计算机视觉领域有哪些成果？"],
            ["最近的研究创新点有哪些？"],
            ["介绍一下图像描述生成技术"],
        ],
        inputs=question_input
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 设为True可以生成公网链接
        inbrowser=True
    )