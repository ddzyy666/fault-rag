'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { Button } from '@/components/ui/button';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from '@/components/ui/sheet';

type Run = {
  id: string;
  question: string;
  status: string;
  phase: string;
  created_at: string;
  elapsed_ms: number | null;
  model_name: string;
  model_turns: number;
  usage: Record<string, number | null>;
  failure_reason: string | null;
};
type ToolCall = {
  id: string;
  sequence: number;
  step: number;
  tool_name: string;
  status: string;
  arguments: Record<string, unknown>;
  result_summary: Record<string, unknown>;
  elapsed_ms: number | null;
};
type Detail = Run & { tool_calls: ToolCall[] };
const labels: Record<string, string> = {
  running: '执行中',
  succeeded: '已完成',
  failed: '失败',
  cancelled: '已取消',
  interrupted: '执行中断',
  ok: '成功',
  not_found: '未找到',
  no_results: '无检索结果',
  forbidden: '范围不允许',
  invalid_arguments: '参数错误',
  timeout: '超时',
  error: '工具失败',
  preparing: '准备上下文',
  model: '模型决策',
  tool: '执行工具',
  persisting: '保存回答',
  completed: '完成',
  get_device_history: '查询设备历史',
  search_manual: '检索维修手册',
};
const duration = (value: number | null) =>
  value === null ? '未知' : `${(value / 1000).toFixed(2)} 秒`;

export function AgentRuns({
  conversationId,
  messageId,
  onClose,
}: {
  conversationId: string;
  messageId?: string;
  onClose: () => void;
}) {
  const [page, setPage] = useState(1);
  const [version, setVersion] = useState(0);
  const [runs, setRuns] = useState<Run[]>([]);
  const [total, setTotal] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState('');
  const [detailError, setDetailError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    const query = new URLSearchParams({ page: String(page), page_size: '20' });
    if (messageId) query.set('assistant_message_id', messageId);
    api<{ items: Run[]; total: number }>(
      `/conversations/${conversationId}/runs?${query}`,
      {
        signal: controller.signal,
      },
    )
      .then((result) => {
        if (controller.signal.aborted) return;
        setRuns(result.items);
        setTotal(result.total);
        setError('');
        setLoading(false);
        if (messageId && result.items.length) setSelected(result.items[0].id);
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) {
          setError(cause instanceof Error ? cause.message : '记录加载失败');
          setLoading(false);
        }
      });
    return () => controller.abort();
  }, [conversationId, messageId, page, version]);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    api<Detail>(`/conversations/${conversationId}/runs/${selected}`, {
      signal: controller.signal,
    })
      .then((result) => {
        if (!controller.signal.aborted) {
          setDetail(result);
          setDetailError('');
        }
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted)
          setDetailError(
            cause instanceof Error ? cause.message : '详情加载失败',
          );
      });
    return () => controller.abort();
  }, [conversationId, selected, version]);

  function refresh() {
    setLoading(true);
    setDetail(null);
    setDetailError('');
    setVersion((value) => value + 1);
  }

  return (
    <Sheet
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <SheetContent className="w-full border-white/10 bg-[#0d1929] text-slate-100 sm:max-w-2xl">
        <SheetHeader className="border-b border-white/10 p-5">
          <SheetTitle className="text-slate-100">
            {messageId ? '查看执行过程' : '会话执行记录'}
          </SheetTitle>
          <SheetDescription className="text-slate-400">
            查询工具、状态和耗时；失败执行也会保留。
          </SheetDescription>
          <Button variant="outline" className="mt-2 w-fit" onClick={refresh}>
            刷新记录
          </Button>
        </SheetHeader>
        <div className="overflow-y-auto p-5 space-y-4">
          {error && (
            <p role="alert" className="text-red-300">
              {error}
            </p>
          )}
          {loading ? (
            <p>正在加载…</p>
          ) : runs.length === 0 && !error ? (
            <p className="text-slate-400">
              暂无执行记录。历史回答和普通 RAG 回答可能未记录执行过程。
            </p>
          ) : null}
          {!loading &&
            runs.map((run) => (
              <button
                key={run.id}
                onClick={() => {
                  setSelected(run.id);
                  setDetail(null);
                  setDetailError('');
                  if (selected === run.id) setVersion((v) => v + 1);
                }}
                className={`block w-full rounded-xl border p-3 text-left ${selected === run.id ? 'border-cyan-400/60' : 'border-white/10'}`}
              >
                <div className="flex justify-between gap-3 text-sm">
                  <span className="min-w-0 break-words">{run.question}</span>
                  <span className="shrink-0 text-cyan-300">
                    {labels[run.status] ?? run.status}
                  </span>
                </div>
                <p className="mt-2 text-xs text-slate-400">
                  {new Date(run.created_at).toLocaleString()} ·{' '}
                  {duration(run.elapsed_ms)}
                </p>
              </button>
            ))}
          {total > 20 && (
            <div className="flex items-center justify-between">
              <Button
                disabled={loading || page === 1}
                onClick={() => {
                  setLoading(true);
                  setPage(page - 1);
                }}
              >
                上一页
              </Button>
              <span>
                {page} / {Math.ceil(total / 20)}
              </span>
              <Button
                disabled={loading || page * 20 >= total}
                onClick={() => {
                  setLoading(true);
                  setPage(page + 1);
                }}
              >
                下一页
              </Button>
            </div>
          )}
          {selected && !detail && !detailError && <p>正在加载执行详情…</p>}
          {detailError && (
            <p role="alert" className="text-red-300">
              {detailError}
            </p>
          )}
          {detail && (
            <section className="border-t border-white/10 pt-4 space-y-4">
              <h3 className="font-semibold">
                执行详情 · {labels[detail.status] ?? detail.status}
              </h3>
              <p className="text-sm break-words">{detail.question}</p>
              <p className="text-xs text-slate-400 break-words">
                模型：{detail.model_name} · 已返回决策 {detail.model_turns} 次 ·
                总耗时 {duration(detail.elapsed_ms)}
              </p>
              <p className="text-xs text-slate-400">
                Token：输入 {detail.usage.prompt_tokens ?? '未知'} / 输出{' '}
                {detail.usage.completion_tokens ?? '未知'} / 合计{' '}
                {detail.usage.total_tokens ?? '未知'}
              </p>
              <p className="text-xs text-slate-400">
                用量为已收到的模型统计，中断请求可能未返回用量。
              </p>
              {detail.failure_reason && (
                <p
                  role="alert"
                  className="rounded-lg bg-red-400/10 p-3 text-sm text-red-200"
                >
                  {labels[detail.phase] ?? detail.phase}：
                  {detail.failure_reason}
                </p>
              )}
              {detail.tool_calls.length === 0 && (
                <p className="text-sm text-slate-400">
                  本轮没有工具执行记录，模型可能直接回答、追问或在调用前失败。
                </p>
              )}
              {detail.tool_calls.map((call) => (
                <article
                  key={call.id}
                  className="rounded-xl border border-white/10 p-3 space-y-2"
                >
                  <h4 className="text-sm font-medium">
                    {call.sequence}. {labels[call.tool_name] ?? call.tool_name}
                  </h4>
                  <p className="text-xs text-slate-400">
                    决策轮次 {call.step} · {labels[call.status] ?? call.status}{' '}
                    · {duration(call.elapsed_ms)}
                  </p>
                  <details className="text-xs text-slate-300">
                    <summary className="cursor-pointer">调用参数</summary>
                    <pre className="mt-2 whitespace-pre-wrap break-all">
                      {JSON.stringify(call.arguments, null, 2)}
                    </pre>
                  </details>
                  <details className="text-xs text-slate-300" open>
                    <summary className="cursor-pointer">结果摘要</summary>
                    <pre className="mt-2 whitespace-pre-wrap break-all">
                      {JSON.stringify(call.result_summary, null, 2)}
                    </pre>
                  </details>
                </article>
              ))}
            </section>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
