import nltk
import ssl

# --- 核心改动：指定国内镜像源 ---
# 这是一个托管在GitHub上的常用NLTK数据镜像
# 它利用了GitHub的CDN，在国内访问速度通常比官方源快得多
nltk.data.path.append('./nltk_data/') # 建议下载到当前目录下的nltk_data文件夹
nltk.download('punkt', download_dir='./nltk_data/')
nltk.download('averaged_perceptron_tagger', download_dir='./nltk_data/')
nltk.download('maxent_ne_chunker', download_dir='./nltk_data/') # 顺便下载命名实体识别
nltk.download('words', download_dir='./nltk_data/') # 词汇表

# 解决SSL证书问题，这是一个好习惯
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- 指定下载的包列表 ---
packages_to_download = [
    'punkt',                    # 用于分句
    'averaged_perceptron_tagger', # 用于词性标注 (这次报错的根源)
    'maxent_ne_chunker',        # 用于命名实体识别 (unstructured可能用到)
    'words'                     # 常用词列表 (unstructured可能用到)
]

# --- 执行下载 ---
for package in packages_to_download:
    try:
        print(f"--- 正在从镜像源下载: {package} ---")
        nltk.download(package)
        print(f"--- {package} 下载成功！ ---")
    except Exception as e:
        print(f"!!! 下载 {package} 失败: {e} !!!")
        print("请检查网络连接或尝试手动下载。")

print("\n所有必要的NLTK数据包已尝试下载完毕。")