import os
import re
import hashlib
import pymupdf
import chromadb

from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"



from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-embedding:4b")

PDF_PATH = r"H5U_PLC_Manual.pdf"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 80
BATCH_SIZE = 100

# 定义两个全局列表，用来最后盖章使用
total_new_ids = []              # 用来存放 id
total_new_metas = []            # 用来存放元数据

# 计算问文件本体二进制 MD5，只看文件内容，忽略路径、文件名
def calc_file_binary_md5(file_path) -> str:
    hash_obj = hashlib.md5()
    # 分块读取，避免一次性加载超大文件内存（很重要）
    with open(file_path, "rb") as f:
        while chunk := f.read(4096 * 8):    # 每次读取 32KB,:= 新语法，可以同时表示为真，也就是读出的数据不为空
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

print("1. 正在初始化环境和数据库...")

# ========= 旧代码，从网络下载向量模型===================
# # 初始化向量模型
# model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
# ==================================================

# ========= Ollama 本地加载向量模型====================
ollama_ef = OllamaEmbeddingFunction(
    url = OLLAMA_BASE_URL,
    model_name = MODEL_NAME          # 使用本地模型
)

# 创建向量数据库
client = chromadb.PersistentClient(path="./chroma_db")

# 创建数据表对象 collection，名称为 plc_knowledge 关键点：我们强制指定使用余弦距离（cosine），这样以后算出来的距离就是 0~1 之间了
collection = client.get_or_create_collection(
    name="plc_knowledge",
    metadata={"hnsw:space": "cosine"},
    embedding_function=ollama_ef                        # 挂载 ollama_ef
)


# ================== 第一次去重：文件级预捡去重 =================

# 调用生成哈希值（方法），将文件加指纹
file_fingerprint = calc_file_binary_md5(PDF_PATH)
print(f"文件指纹：{file_fingerprint}")

# 清理上次崩溃残留的processing碎片，当写入向量库遇上突发情况，比如断电之类的，导致写入只进行了一半，需要把向量库中残余的不全向量删除
# 根据条件进行删除，where 条件，键为 file_id 的，并且状态 status 为 processing 的
# cllection 查询，delete 删除
collection.delete(
    where={
        "$and":[                            # 两个条件需要用 $and 运算符
            {"file_id":file_fingerprint},
            {"status":"processing"}
        ]
    }
)

# 去库里查一下有没有这个指纹，collection 是数据表，条件是 file_id，状态是 finished ，文件完成全部写入，所有写入的 chunk 的 status 更新为 finished
# 这里的 limit = 1 表示只查询到 1 条就结束查询，因为每个 chunk 都会有 file_id 状态，所以只要出现一个，并且 status 为 finished 就说明该文件已被完整写入向量库
existing_file = collection.get(
    where={
        "$and":[
            {"file_id": file_fingerprint},
            {"status":"finished"}
        ]
    },
    limit=1
)

if existing_file["ids"]:
    print("⚠️ 库里已经存在该文件，无需重复解析！退出。")
    exit()      # 退出，终止全部程序，实际上是抛出了 SystemExit 的异常，python解释器捕获了该异常就会关掉自己，生产级代码会 import sys 调用 sys.exit()，exit(0)正常退出，exit(1)异常退出

print("✅ 库里没有该文件，开始解析...")
#===========================================================

print("2. 正在提取全文（带隐形页码标签）...")
# 创建 doc 对象，接受 pymupdf.open() 返回的 document
doc = pymupdf.open(PDF_PATH)
full_text = ""

# 遍历 doc ，enumerate() 为 python 内置函数（枚举），会自动翻页并且计数，返回一个带编码的列表，并解包为 页码 和 页内容
for page_num, page in enumerate(doc):   # type:ignore
    text = page.get_text("text")

    # ==================== 核心清洗环节 ====================
    print("提取完成，正在清洗无用字符...")
    # 1. 把连续的空格（2个及以上）替换成一个空格，[ \t]表示一个箱子里既有 空格 也有 制表符，+ 表示一个或一个以上
    text = re.sub(r'[ \t]+', ' ', text)

    # 2. 把连续的换行（3个及以上）替换成两个换行（保留段落感），{3，}表示3个起步，上不封顶
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 3. 去掉一些 PDF 解析常出现的不可见字符（比如软连接字符），.replace(旧内容，新内容)查旧换新，\u00ad 软连接，\xa0不间断空格
    # 链式调用，先执行第一个，再执行第二个，然后返回给 text
    text = text.replace('\u00ad', '').replace('\xa0', ' ')

    if not text.strip():
        continue

    # 在每页开头打上隐形标签（例如：【第5页】），用于后续提取
    full_text += f"\n【第{page_num + 1}页】\n" + text
    # =====================================================

print("3. 正在智能切分文本...")
# 设置 Rcursive 递归切法，600字内会先找段落\n\n，找不到找换行\n，以此类推...
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,                 # 块大小：切分时最大字符数，根据文档不同，设置不同大小，设置太小容易打断，设置太大容易抓不住重点
    chunk_overlap=CHUNK_OVERLAP,               # 块重叠：相邻块之间，重叠的字符数，一般设为块大小的10%或15%
    length_function=len,                    # Python 底层 len() 方法，表示切分时直接使用底层的 len() 方法计算长度
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]  # 切分优先级
)
# 切分，返回一个列表给 chunks，切分后的块集合
chunks = text_splitter.split_text(full_text)
print(f"切分成了 {len(chunks)} 个 chunk。")

print("4. 正在生成向量并写入数据库...")
# 分批处理。防止一次性把内存撑爆，100个块为 1 批
for i in range(0, len(chunks), BATCH_SIZE):
    batch_texts = chunks[i:i + BATCH_SIZE]          # 拆分 100 个块
    batch_metas = []

    # 遍历 100 个块，把每个块的页码找出来，制作batch_metas标签
    for text in batch_texts:
        pages = re.findall(r'【第(\d+)页】', text)      # \d 表示数字，+ 一个或一个以上
        page_str = f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else (pages[0] if pages else "未知")   # pages[-1]代表倒数最后一个元素
        batch_metas.append({
            "filename": os.path.basename(PDF_PATH),  # 只保留文件名，五路径
            "page":page_str,
            "file_id": file_fingerprint,
            "status": "processing"
        })
    # 删除 100 个块原文中的页码，生成一个没有页码的 100 个块
    clean_batch_texts = [re.sub(r'【第\d+页】\n?', '', t) for t in batch_texts]     # ？表示前面\n可有可无

    # ========================真正的 Chunk 级去重=============================
    # 1. 先算出所有文本的 MD5，遍历清除过页码的 100 个块中的每个块，单独生成 MD5 校验码
    all_ids = [hashlib.md5(t.encode('utf-8')).hexdigest() for t in clean_batch_texts]       # hashlib.md5() 只能对二进制字节进行编码，转换成hex十六进制，存入all_ids列表中

    # 2. 去库里查一下，那些 ID 已经存在
    existing_data = collection.get(ids=all_ids)         # 已经存在的 哈希 列表
    existing_ids = set(existing_data["ids"])            # 把上一步拿到的重复 哈希 的列表转换成集合，后期好比对

    # 3. 把不需要的过滤掉，只保留库里没有的新文本
    new_texts = []
    new_metas = []
    new_ids = []

    # 遍历 100 个块的 哈希列表，用枚举的方式编号
    for j,chunk_id in enumerate(all_ids):
        if chunk_id not in existing_ids:                # 判断当前枚举到的 哈希值 是否在重复 哈希值 列表中，如果不存在，则写入到更新的列表中，这里就用到集合existing_ids，时间复杂度低
            new_texts.append(clean_batch_texts[j])      # 不重复的文本列表
            new_metas.append(batch_metas[j])            # 不重复的标签列表
            new_ids.append(chunk_id)                    # 不重复的 哈希值

    if not new_texts:
        print(f"批次{i} - {i+len(batch_texts)}：全部已存在，跳过向量计算！")

    print(f"批次 {i} - {i+len(batch_texts)}：发现 {len(new_texts)} 个新快，正在计算向量...")

    # # 4. 只对真正的新闻本算向量并入库
    # embeddings = model.encode(new_texts).tolist()       # .encode(new_texts) 返回的是一个矩阵，但向量库只认列表，所以用 .tolist()

    # 批量存入

    collection.add(
        ids=new_ids,                    #
        documents=new_texts,            # 存放原文
        # embeddings=embeddings,        # 存放向量，原来是手动写向量，现在直接挂载了ollama本地向量，直接传文本就可以
        metadatas=new_metas             # 名词叫元数据，用来存放自定义的标签之类的
    )

    total_new_ids.extend(new_ids)
    total_new_metas.extend(new_metas)
    # =========================================================================
# 5. 所有批次都成功写完后，开始“盖章”
print("5. 正在更新文档状态为已完成...")

if not total_new_ids:
    print("本次没有新增任何数据块，无需盖章。")
else:
    # 利用前面收集的全局变量，进行分批 Update
    for i in range(0, len(total_new_ids), BATCH_SIZE):
        batch_ids = total_new_ids[i:i+BATCH_SIZE]           # 抓取 BATCH_SIZE 个数据，这里是100
        batch_metas = total_new_metas[i:i+BATCH_SIZE]

        # 复制并修改状态
        updated_metadatas = []
        for meta in batch_metas:
            new_meta = meta.copy()                          # 复制
            new_meta["status"] = "finished"                 # 更改状态，用键更改
            updated_metadatas.append(new_meta)              # 放到新列表当中

        # 提交更新
        collection.update(
            ids=batch_ids,                                  # 按照主键去更新
            metadatas=updated_metadatas
        )
        print(f"已盖章 {i + len(batch_ids)} / {len(total_new_ids)} 个块状态...")
    print("✅ 数据库状态已全部更新为：finished")

# 最后一步
print("✅ PLC 知识库构建完成！")
