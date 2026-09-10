# Agent 执行记录

## 查看方式

更新后端数据库迁移并重启服务。在聊天中完成一次 Agent 问答，点击回答下方“查看执行过程”。
刷新页面、重新打开会话后仍可查询；会话顶部“执行记录”包含没有生成回答的失败轮次。
记录列表支持分页，执行中可点“刷新记录”。老消息和普通 RAG 回答没有执行记录。

## 保存的数据

- `agent_runs`：会话 ID、脱敏后的本轮问题、模型名、执行状态、最后阶段、开始/结束时间、
  总耗时、已返回的模型决策数、累计 Token 用量、失败原因、成功回答 ID。
- `agent_tool_calls`：本轮执行顺序、决策轮次、调用 ID、工具名、允许字段内的参数、
  结果状态、摘要、开始/结束时间和耗时。
- 查设备只记录设备编号、返回记录条数和模拟标记；检索只记录切片 ID、数量和重排状态。
  完整维修资料仍由原有回答引用保存；不复制到执行记录，不保存模型内部推理。
- 配置中的模型/重排/Qdrant 密钥、常见 Bearer/sk- 密钥及命名的 token/password 等文本会脱敏。
  工具参数仅保存预期字段，未知字段省略；无效 JSON 只保存标记。此策略不是任意敏感信息识别器。
  该脱敏策略针对新增执行记录，不改写原有聊天消息正文。

## 事务和状态

执行与工具事件使用独立会话、短事务提交，使用与业务会话相同的数据库绑定。因此模型失败、
工具 rollback 或消息保存失败不会撤销已提交的执行记录。
成功保存回答时，在同一事务内写两条消息、引用及 `assistant_message_id`，并将执行置为
`succeeded`。关联失败时消息也回滚，避免“成功但找不到回答”。

状态为 `running`、`succeeded`、`failed`、`cancelled`、`interrupted`。
工具有自身的 `ok`、`not_found`、`no_results`、`invalid_arguments`、`timeout` 等状态；
工具失败不等于整轮失败，模型仍可以解释错误或追问。
总超时/模型异常记 `failed`；生成器关闭或请求取消记 `cancelled`，取消清理阶段屏蔽作用域取消，
提交失败记录。未完成工具也会结束，不保存半轮消息。

进程强制退出不能执行清理。查询记录时，超过配置的 Agent 总时限加 60 秒收尾窗口、
仍为 running 的轮次标为 interrupted；不会伪造实际结束时间和耗时。已完成记录不被覆盖。
这不是恢复执行功能；数据库本身不可用时无法保证写入，应在恢复后查看未结束的记录。

Token 用量为已收到的模型统计。若某次返回缺少统计，相应累计字段保持未知；
已发出但中断的请求可能产生费用却未返回统计，页面数字不等同于账单。
删除会话会级联删除执行与工具记录。接口按会话 ID 校验记录归属，现有项目仍未实现用户鉴权。

## 接口

```text
GET /api/v1/conversations/{conversation_id}/runs?page=1&page_size=20
GET /api/v1/conversations/{conversation_id}/runs?assistant_message_id={message_id}
GET /api/v1/conversations/{conversation_id}/runs/{run_id}
```

SSE 的 agent_started、completed 事件含 run_id；完成后可通过关联回答查询详情。
失败即使没有收到 SSE 事件也可从会话执行列表中查找。

## 验证

```powershell
python -m pytest backend/tests/test_agent_runs.py backend/tests/test_agent.py backend/tests/test_device_migration.py
```

测试覆盖成功关联、跨会话隔离、失败记录及已发生的工具/用量保留、消息保存回滚、
关闭流、敏感字段过滤、过期记录处理和级联删除。
