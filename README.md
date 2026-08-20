# 智能故障诊断助手

基于 RAG 的设备故障诊断系统。项目将逐步支持维修手册解析、混合检索、引用溯源、
多轮故障诊断和结构化排查建议。

## 当前功能

- FastAPI 后端基础结构
- 环境变量配置
- 统一 API 响应格式
- 统一异常处理
- 健康检查接口
- SQLAlchemy 2 异步数据库访问
- Alembic 数据库版本迁移
- 知识库创建、分页、详情、更新和删除接口
- 知识库、文档、切片、会话和消息核心数据模型
- PDF、DOCX、TXT、Markdown 文件上传与文本解析
- 文件大小与扩展名校验、SHA-256 内容去重、安全文件名和隔离存储
- 文档解析状态、按页原文查看和失败原因记录
- 接口与数据库隔离测试

## 数据模型

```mermaid
erDiagram
    KNOWLEDGE_BASE ||--o{ DOCUMENT : contains
    DOCUMENT ||--o{ DOCUMENT_PAGE : extracts
    DOCUMENT ||--o{ DOCUMENT_CHUNK : splits_into
    KNOWLEDGE_BASE o|--o{ CONVERSATION : scopes
    CONVERSATION ||--o{ MESSAGE : contains
```

- 删除知识库时，其文档和切片会级联删除。
- 删除知识库后，历史诊断会话仍然保留，知识库关联被置空。
- 删除会话时，其消息会级联删除。

## 本地运行

```powershell
conda activate fault-rag
Set-Location "D:\python project\rag"
python -m pip install -r backend/requirements-dev.txt
python -m alembic -c backend/alembic.ini upgrade head
python -m uvicorn app.main:app --app-dir backend --reload
```

启动后访问：

- Swagger: <http://127.0.0.1:8000/docs>
- 健康检查: <http://127.0.0.1:8000/api/v1/health>

## 知识库接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/v1/knowledge-bases` | 创建知识库 |
| `GET` | `/api/v1/knowledge-bases` | 分页查询知识库 |
| `GET` | `/api/v1/knowledge-bases/{id}` | 查询知识库详情 |
| `PATCH` | `/api/v1/knowledge-bases/{id}` | 更新知识库 |
| `DELETE` | `/api/v1/knowledge-bases/{id}` | 删除知识库 |

## 文档接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/v1/knowledge-bases/{id}/documents` | 上传并解析文档 |
| `GET` | `/api/v1/knowledge-bases/{id}/documents` | 分页查询知识库文档 |
| `GET` | `/api/v1/documents/{id}` | 查询文档详情与处理状态 |
| `GET` | `/api/v1/documents/{id}/content` | 分页查看按页解析文本 |
| `DELETE` | `/api/v1/documents/{id}` | 删除文档、原始文件和解析内容 |

支持的文件格式：

- PDF：使用 PyMuPDF 按页提取文本
- DOCX：使用 python-docx 提取段落和表格文本
- TXT、Markdown：支持 UTF-8、UTF-8 BOM 和 GB18030 编码

默认文件大小上限为 20 MB，可通过 `MAX_UPLOAD_SIZE_BYTES` 修改。原始文件存放在
`UPLOAD_DIR` 指定目录，同一知识库内按 SHA-256 内容哈希去重。扫描版 PDF 当前不会
执行 OCR，会被标记为 `failed` 并记录失败原因。

文档处理状态：

```text
pending → parsing → parsed → indexed
                    └→ failed
```

当前上传请求会同步完成文本解析。后续生产化阶段可以将 `parsing` 部分迁移到任务队列，
接口、数据库状态和解析服务不需要重新设计。

当前默认使用项目根目录下的 SQLite 数据库 `fault_rag.db`。该文件已被 Git 忽略，
后续部署阶段会通过 `DATABASE_URL` 切换到 PostgreSQL。

## 数据库迁移

应用已有迁移：

```powershell
python -m alembic -c backend/alembic.ini current
python -m alembic -c backend/alembic.ini history
```

修改 ORM 模型后创建并应用迁移：

```powershell
python -m alembic -c backend/alembic.ini revision --autogenerate -m "describe change"
python -m alembic -c backend/alembic.ini upgrade head
```

## 测试与代码检查

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
```
