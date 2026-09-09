'use client';

import {
  AlertCircle,
  BookOpen,
  Bot,
  Boxes,
  CheckCircle2,
  ChevronRight,
  FileText,
  LoaderCircle,
  MessageSquareText,
  Plus,
  RefreshCw,
  Send,
  Upload,
  UserRound,
  Wrench,
} from 'lucide-react';
import {
  type SyntheticEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import { Textarea } from '@/components/ui/textarea';
import {
  API_BASE,
  api,
  ChatMessage,
  Conversation,
  DocumentItem,
  KnowledgeBase,
  parseSse,
  Source,
} from '@/lib/api';

type ListResult<T> = { items: T[]; total: number };
const quickQuestions = [
  '空压机出现 E101 高温停机，应该先检查什么？',
  '设备持续加载但压力上不去，可能有哪些原因？',
  '维修空压机前需要采取哪些安全措施？',
];
const statusLabels: Record<string, string> = {
  pending: '等待处理',
  parsing: '解析中',
  parsed: '已解析',
  chunking: '分块中',
  chunked: '已分块',
  indexing: '索引中',
  indexed: '可检索',
  failed: '处理失败',
};
const textValue = (value: unknown, fallback = '') =>
  typeof value === 'string' ? value : fallback;

export default function Home() {
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [activeKb, setActiveKb] = useState<KnowledgeBase | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversation, setActiveConversation] =
    useState<Conversation | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [question, setQuestion] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  const [documentOpen, setDocumentOpen] = useState(false);
  const [sourceOpen, setSourceOpen] = useState(false);
  const [selectedSources, setSelectedSources] = useState<Source[]>([]);
  const [newKbName, setNewKbName] = useState('');
  const [newKbDescription, setNewKbDescription] = useState('');
  const [busyDocument, setBusyDocument] = useState('');
  const chatEnd = useRef<HTMLDivElement>(null);

  const loadKnowledgeBases = useCallback(async () => {
    try {
      const result = await api<ListResult<KnowledgeBase>>(
        '/knowledge-bases?page_size=100',
      );
      setKnowledgeBases(result.items);
      setActiveKb((current) => current ?? result.items[0] ?? null);
      setError('');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '无法连接后端服务');
    } finally {
      setLoading(false);
    }
  }, []);

  const loadWorkspace = useCallback(async (knowledgeBaseId: string) => {
    try {
      const [conversationResult, documentResult] = await Promise.all([
        api<ListResult<Conversation>>(
          `/conversations?knowledge_base_id=${knowledgeBaseId}&page_size=100`,
        ),
        api<ListResult<DocumentItem>>(
          `/knowledge-bases/${knowledgeBaseId}/documents?page_size=100`,
        ),
      ]);
      setConversations(conversationResult.items);
      setDocuments(documentResult.items);
      setError('');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '工作区加载失败');
    }
  }, []);

  useEffect(() => {
    // oxlint-disable-next-line react/react-compiler -- initial API synchronization
    void loadKnowledgeBases();
  }, [loadKnowledgeBases]);
  useEffect(() => {
    if (activeKb) {
      // oxlint-disable-next-line react/react-compiler -- synchronize selected workspace
      void loadWorkspace(activeKb.id);
    }
  }, [activeKb, loadWorkspace]);
  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function selectConversation(conversation: Conversation) {
    setActiveConversation(conversation);
    try {
      const result = await api<ListResult<ChatMessage>>(
        `/conversations/${conversation.id}/messages?page_size=100`,
      );
      setMessages(result.items);
      setError('');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '会话加载失败');
    }
  }

  function selectKnowledgeBase(knowledgeBase: KnowledgeBase) {
    setActiveKb(knowledgeBase);
    setActiveConversation(null);
    setMessages([]);
  }

  async function createKnowledgeBase(event: SyntheticEvent<HTMLFormElement>) {
    event.preventDefault();
    try {
      const created = await api<KnowledgeBase>('/knowledge-bases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newKbName,
          description: newKbDescription || null,
        }),
      });
      setKnowledgeBases((items) => [created, ...items]);
      selectKnowledgeBase(created);
      setNewKbName('');
      setNewKbDescription('');
      setCreateOpen(false);
      setError('');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '创建知识库失败');
    }
  }

  async function uploadDocument(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file || !activeKb) return;
    setBusyDocument('upload');
    try {
      const form = new FormData();
      form.append('file', file);
      await api<DocumentItem>(`/knowledge-bases/${activeKb.id}/documents`, {
        method: 'POST',
        body: form,
      });
      await loadWorkspace(activeKb.id);
      setError('');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '文档上传失败');
    } finally {
      setBusyDocument('');
      event.target.value = '';
    }
  }

  async function advanceDocument(document: DocumentItem) {
    const action = document.status === 'parsed' ? 'chunks' : 'index';
    setBusyDocument(document.id);
    try {
      await api(`/documents/${document.id}/${action}`, {
        method: 'POST',
        headers:
          action === 'chunks'
            ? { 'Content-Type': 'application/json' }
            : undefined,
        body:
          action === 'chunks'
            ? JSON.stringify({
                chunk_size: 700,
                chunk_overlap: 100,
                min_chunk_size: 80,
              })
            : undefined,
      });
      if (activeKb) await loadWorkspace(activeKb.id);
      setError('');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '文档处理失败');
    } finally {
      setBusyDocument('');
    }
  }

  async function ensureConversation() {
    if (activeConversation) return activeConversation;
    if (!activeKb) throw new Error('请先选择知识库');
    const created = await api<Conversation>('/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ knowledge_base_id: activeKb.id, title: '新诊断' }),
    });
    setActiveConversation(created);
    setConversations((items) => [created, ...items]);
    return created;
  }

  async function sendQuestion(event?: SyntheticEvent<HTMLFormElement>) {
    'use no memo';
    event?.preventDefault();
    const content = question.trim();
    if (!content || sending) return;
    setQuestion('');
    setSending(true);
    setError('');
    const stamp = Date.now();
    const userId = `user-${stamp}`;
    const assistantId = `assistant-${stamp}`;
    setMessages((items) => [
      ...items,
      { id: userId, role: 'user', content, citations: [] },
      {
        id: assistantId,
        role: 'assistant',
        content: '',
        citations: [],
        streaming: true,
      },
    ]);
    try {
      const conversation = await ensureConversation();
      const response = await fetch(
        `${API_BASE}/conversations/${conversation.id}/messages/stream`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: content,
            top_k: 5,
            score_threshold: 0.3,
            retrieval_mode: 'hybrid',
            rerank: true,
          }),
        },
      );
      await parseSse(response, (eventName, data) => {
        if (eventName === 'sources') {
          const sources = (data.items ?? []) as Source[];
          setMessages((items) =>
            items.map((item) =>
              item.id === assistantId ? { ...item, citations: sources } : item,
            ),
          );
        }
        if (eventName === 'answer_delta') {
          const delta = textValue(data.delta);
          setMessages((items) =>
            items.map((item) =>
              item.id === assistantId
                ? { ...item, content: item.content + delta }
                : item,
            ),
          );
        }
        if (eventName === 'completed')
          setMessages((items) =>
            items.map((item) =>
              item.id === assistantId
                ? {
                    ...item,
                    id: textValue(data.assistant_message_id, item.id),
                    streaming: false,
                  }
                : item,
            ),
          );
        if (eventName === 'error')
          throw new Error(textValue(data.message, '诊断生成失败'));
      });
      if (activeKb) await loadWorkspace(activeKb.id);
    } catch (cause) {
      const errorMessage =
        cause instanceof Error ? cause.message : '诊断生成失败';
      setError(errorMessage);
      setMessages((items) =>
        items.map((item) =>
          item.id === assistantId
            ? {
                ...item,
                content: item.content || '生成失败，请查看上方错误提示。',
                streaming: false,
              }
            : item,
        ),
      );
    } finally {
      setSending(false);
    }
  }

  function openSources(sources: Source[]) {
    setSelectedSources(sources);
    setSourceOpen(true);
  }
  const indexedCount = documents.filter(
    (item) => item.status === 'indexed',
  ).length;

  return (
    <main className="min-h-screen bg-[#07111f] text-slate-100">
      <div className="industrial-grid pointer-events-none fixed inset-0" />
      <div className="relative mx-auto flex min-h-screen max-w-[1680px] flex-col p-3 sm:p-5">
        <header className="mb-3 flex h-16 items-center justify-between rounded-2xl border border-white/10 bg-[#0b1728]/95 px-4 shadow-2xl shadow-black/20 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid size-10 shrink-0 place-items-center rounded-xl bg-cyan-400 text-[#07111f] shadow-[0_0_28px_rgba(34,211,238,.28)]">
              <Wrench className="size-5" />
            </div>
            <div className="min-w-0">
              <h1 className="truncate text-base font-semibold tracking-wide sm:text-lg">
                智能故障诊断助手
              </h1>
              <p className="text-xs text-slate-400">知识检索与维修决策工作台</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge className="hidden border-emerald-400/20 bg-emerald-400/10 text-emerald-300 sm:flex">
              <span className="size-1.5 rounded-full bg-emerald-400" /> API
            </Badge>
            <Button
              variant="outline"
              className="h-9 border-white/10 bg-white/5 text-slate-100 hover:bg-white/10"
              onClick={() => setDocumentOpen(true)}
              disabled={!activeKb}
            >
              <FileText /> 文档{' '}
              <span className="text-slate-400">{documents.length}</span>
            </Button>
          </div>
        </header>
        {error && (
          <div className="mb-3 flex items-center gap-2 rounded-xl border border-rose-400/20 bg-rose-400/10 px-4 py-2.5 text-sm text-rose-200">
            <AlertCircle className="size-4 shrink-0" />
            <span className="flex-1">{error}</span>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => setError('')}
              aria-label="关闭提示"
            >
              ×
            </Button>
          </div>
        )}
        <div className="grid min-h-0 flex-1 gap-3 lg:grid-cols-[280px_minmax(0,1fr)] xl:grid-cols-[280px_minmax(0,1fr)_320px]">
          <aside className="flex min-h-[300px] flex-col rounded-2xl border border-white/10 bg-[#0b1728]/92 p-3">
            <div className="mb-2 flex items-center justify-between px-2 py-1">
              <span className="text-xs font-semibold uppercase tracking-[.18em] text-slate-500">
                知识库
              </span>
              <Button
                variant="ghost"
                size="icon-sm"
                className="text-cyan-300 hover:bg-cyan-400/10"
                onClick={() => setCreateOpen(true)}
                aria-label="创建知识库"
              >
                <Plus />
              </Button>
            </div>
            <div className="space-y-1">
              {loading ? (
                <div className="flex items-center gap-2 px-2 py-3 text-sm text-slate-500">
                  <LoaderCircle className="size-4 animate-spin" />
                  加载中
                </div>
              ) : knowledgeBases.length ? (
                knowledgeBases.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => selectKnowledgeBase(item)}
                    className={`group flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition ${activeKb?.id === item.id ? 'bg-cyan-400/12 text-cyan-100 ring-1 ring-cyan-400/20' : 'text-slate-300 hover:bg-white/5'}`}
                  >
                    <Boxes className="size-4 shrink-0" />
                    <span className="min-w-0 flex-1 truncate text-sm font-medium">
                      {item.name}
                    </span>
                    <ChevronRight className="size-3.5 opacity-40" />
                  </button>
                ))
              ) : (
                <button
                  className="w-full rounded-xl border border-dashed border-white/10 p-4 text-left text-sm text-slate-400"
                  onClick={() => setCreateOpen(true)}
                >
                  创建第一个知识库
                </button>
              )}
            </div>
            <div className="my-4 h-px bg-white/8" />
            <div className="mb-2 flex items-center justify-between px-2 py-1">
              <span className="text-xs font-semibold uppercase tracking-[.18em] text-slate-500">
                诊断记录
              </span>
              <Button
                variant="ghost"
                size="icon-sm"
                className="text-cyan-300 hover:bg-cyan-400/10"
                onClick={() => {
                  setActiveConversation(null);
                  setMessages([]);
                }}
                disabled={!activeKb}
                aria-label="新建诊断"
              >
                <Plus />
              </Button>
            </div>
            <div className="min-h-0 flex-1 space-y-1 overflow-y-auto">
              {conversations.map((item) => (
                <button
                  key={item.id}
                  onClick={() => void selectConversation(item)}
                  className={`flex w-full items-start gap-3 rounded-xl px-3 py-2.5 text-left transition ${activeConversation?.id === item.id ? 'bg-white/8 text-white' : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'}`}
                >
                  <MessageSquareText className="mt-0.5 size-4 shrink-0" />
                  <span className="line-clamp-2 text-sm">{item.title}</span>
                </button>
              ))}
            </div>
            <div className="mt-3 rounded-xl border border-white/8 bg-[#07111f]/70 p-3">
              <div className="mb-2 flex items-center justify-between text-xs text-slate-400">
                <span>索引状态</span>
                <span>
                  {indexedCount}/{documents.length}
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-white/8">
                <div
                  className="h-full rounded-full bg-cyan-400"
                  style={{
                    width: `${documents.length ? (indexedCount / documents.length) * 100 : 0}%`,
                  }}
                />
              </div>
            </div>
          </aside>

          <section className="flex min-h-[620px] min-w-0 flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#0d1929]/95">
            <div className="flex items-center justify-between border-b border-white/8 px-5 py-4">
              <div className="min-w-0">
                <h2 className="truncate font-semibold">
                  {activeConversation?.title ?? '新诊断'}
                </h2>
                <p className="mt-0.5 truncate text-xs text-slate-500">
                  {activeKb?.name ?? '请选择知识库'}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Badge
                  variant="outline"
                  className="border-cyan-400/20 text-cyan-300"
                >
                  混合检索
                </Badge>
                <Badge
                  variant="outline"
                  className="hidden border-violet-400/20 text-violet-300 sm:flex"
                >
                  Reranker
                </Badge>
              </div>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-8">
              {!messages.length ? (
                <div className="mx-auto flex h-full max-w-2xl flex-col items-center justify-center text-center">
                  <div className="mb-5 grid size-16 place-items-center rounded-2xl border border-cyan-400/20 bg-cyan-400/8 text-cyan-300">
                    <Bot className="size-8" />
                  </div>
                  <h2 className="text-xl font-semibold">
                    描述设备现象或输入故障码
                  </h2>
                  <p className="mt-2 max-w-md text-sm leading-6 text-slate-400">
                    系统会检索当前知识库，并给出带原文引用的排查建议。
                  </p>
                  <div className="mt-7 grid w-full gap-2 sm:grid-cols-3">
                    {quickQuestions.map((item) => (
                      <button
                        key={item}
                        onClick={() => setQuestion(item)}
                        className="rounded-xl border border-white/8 bg-white/[.025] p-3 text-left text-sm leading-5 text-slate-300 transition hover:border-cyan-400/25 hover:bg-cyan-400/5"
                      >
                        {item}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="mx-auto max-w-3xl space-y-7">
                  {messages.map((chatMessage) => (
                    <article
                      key={chatMessage.id}
                      className={`flex gap-3 ${chatMessage.role === 'user' ? 'justify-end' : ''}`}
                    >
                      {chatMessage.role === 'assistant' && (
                        <div className="grid size-8 shrink-0 place-items-center rounded-lg bg-cyan-400 text-[#07111f]">
                          <Bot className="size-4" />
                        </div>
                      )}
                      <div
                        className={`max-w-[88%] ${chatMessage.role === 'user' ? 'rounded-2xl rounded-tr-sm bg-sky-500 px-4 py-3 text-white' : 'min-w-0 flex-1'}`}
                      >
                        <div className="whitespace-pre-wrap text-[15px] leading-7">
                          {chatMessage.content || (
                            <span className="inline-flex items-center gap-2 text-slate-400">
                              <LoaderCircle className="size-4 animate-spin" />
                              正在检索维修资料
                            </span>
                          )}
                        </div>
                        {chatMessage.role === 'assistant' &&
                          chatMessage.citations.length > 0 && (
                            <button
                              onClick={() => openSources(chatMessage.citations)}
                              className="mt-3 inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-cyan-300 hover:bg-white/8"
                            >
                              <BookOpen className="size-3.5" /> 查看{' '}
                              {chatMessage.citations.length} 条引用资料
                            </button>
                          )}
                      </div>
                      {chatMessage.role === 'user' && (
                        <div className="grid size-8 shrink-0 place-items-center rounded-lg bg-white/10 text-slate-300">
                          <UserRound className="size-4" />
                        </div>
                      )}
                    </article>
                  ))}
                  <div ref={chatEnd} />
                </div>
              )}
            </div>
            <form
              onSubmit={(event) => void sendQuestion(event)}
              className="border-t border-white/8 bg-[#091522] p-3 sm:p-4"
            >
              <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border border-white/10 bg-white/[.035] p-2 focus-within:border-cyan-400/35">
                <Textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                      event.preventDefault();
                      void sendQuestion();
                    }
                  }}
                  placeholder={
                    activeKb
                      ? '输入故障现象、报警码或追问…'
                      : '请先选择或创建知识库'
                  }
                  disabled={!activeKb || sending}
                  className="max-h-36 min-h-11 resize-none border-0 bg-transparent px-3 py-2.5 text-[15px] text-white shadow-none focus-visible:ring-0"
                />
                <Button
                  type="submit"
                  size="icon-lg"
                  disabled={!question.trim() || !activeKb || sending}
                  className="mb-0.5 rounded-xl bg-cyan-400 text-[#07111f] hover:bg-cyan-300"
                >
                  {sending ? (
                    <LoaderCircle className="animate-spin" />
                  ) : (
                    <Send />
                  )}
                  <span className="sr-only">发送</span>
                </Button>
              </div>
              <p className="mt-2 text-center text-[11px] text-slate-600">
                诊断建议仅依据知识库生成，维修前请停机、断电并泄压。
              </p>
            </form>
          </section>

          <aside className="hidden rounded-2xl border border-white/10 bg-[#0b1728]/92 p-5 xl:block">
            <div className="mb-5 flex items-center justify-between">
              <h3 className="font-semibold">知识库概况</h3>
              <Button
                variant="ghost"
                size="icon-sm"
                className="text-slate-400"
                onClick={() => activeKb && void loadWorkspace(activeKb.id)}
              >
                <RefreshCw />
              </Button>
            </div>
            <div className="rounded-xl border border-white/8 bg-white/[.025] p-4">
              <p className="text-xs uppercase tracking-wider text-slate-500">
                当前知识库
              </p>
              <p className="mt-2 font-medium">{activeKb?.name ?? '未选择'}</p>
              <p className="mt-2 text-sm leading-6 text-slate-400">
                {activeKb?.description || '暂无描述'}
              </p>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-3">
              <div className="rounded-xl border border-white/8 bg-white/[.025] p-4">
                <p className="text-2xl font-semibold text-cyan-300">
                  {documents.length}
                </p>
                <p className="mt-1 text-xs text-slate-500">文档总数</p>
              </div>
              <div className="rounded-xl border border-white/8 bg-white/[.025] p-4">
                <p className="text-2xl font-semibold text-emerald-300">
                  {indexedCount}
                </p>
                <p className="mt-1 text-xs text-slate-500">可检索</p>
              </div>
            </div>
            <div className="mt-6">
              <p className="mb-3 text-xs font-semibold uppercase tracking-[.18em] text-slate-500">
                处理流程
              </p>
              {['文档解析', '语义分块', '向量索引', '混合检索与重排'].map(
                (item, index) => (
                  <div
                    key={item}
                    className="relative flex gap-3 pb-5 last:pb-0"
                  >
                    {index < 3 && (
                      <span className="absolute left-[7px] top-4 h-full w-px bg-white/10" />
                    )}
                    <CheckCircle2 className="relative z-10 mt-0.5 size-4 shrink-0 text-cyan-400" />
                    <p className="text-sm text-slate-300">{item}</p>
                  </div>
                ),
              )}
            </div>
          </aside>
        </div>
      </div>

      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className="border-white/10 bg-[#0d1929] text-slate-100 sm:max-w-md">
          <form onSubmit={(event) => void createKnowledgeBase(event)}>
            <DialogHeader>
              <DialogTitle>创建知识库</DialogTitle>
              <DialogDescription className="text-slate-400">
                按设备或维修领域组织资料。
              </DialogDescription>
            </DialogHeader>
            <div className="my-5 space-y-4">
              <label htmlFor="kb-name" className="block text-sm">
                <span className="mb-2 block text-slate-300">名称</span>
              </label>
              <Input
                id="kb-name"
                value={newKbName}
                onChange={(event) => setNewKbName(event.target.value)}
                placeholder="例如：空压机维修知识库"
                className="h-10 border-white/10 bg-white/5"
              />
              <label htmlFor="kb-description" className="block text-sm">
                <span className="mb-2 block text-slate-300">说明</span>
              </label>
              <Textarea
                id="kb-description"
                value={newKbDescription}
                onChange={(event) => setNewKbDescription(event.target.value)}
                placeholder="资料范围、设备型号等"
                className="border-white/10 bg-white/5"
              />
            </div>
            <DialogFooter className="border-white/10 bg-white/[.025]">
              <Button
                type="submit"
                disabled={!newKbName.trim()}
                className="bg-cyan-400 text-[#07111f] hover:bg-cyan-300"
              >
                创建
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      <Sheet open={documentOpen} onOpenChange={setDocumentOpen}>
        <SheetContent className="w-full border-white/10 bg-[#0d1929] text-slate-100 sm:max-w-xl">
          <SheetHeader className="border-b border-white/8 p-5">
            <SheetTitle className="text-slate-100">知识库文档</SheetTitle>
            <SheetDescription className="text-slate-400">
              上传后依次完成分块和向量索引。
            </SheetDescription>
          </SheetHeader>
          <div className="p-5">
            <label className="flex cursor-pointer items-center justify-center gap-2 rounded-xl border border-dashed border-cyan-400/25 bg-cyan-400/5 p-5 text-sm text-cyan-200 hover:bg-cyan-400/10">
              {busyDocument === 'upload' ? (
                <LoaderCircle className="size-4 animate-spin" />
              ) : (
                <Upload className="size-4" />
              )}
              上传 PDF、DOCX、TXT 或 Markdown
              <input
                type="file"
                className="sr-only"
                accept=".pdf,.docx,.txt,.md"
                onChange={(event) => void uploadDocument(event)}
                disabled={busyDocument === 'upload'}
              />
            </label>
            <div className="mt-5 space-y-3">
              {documents.map((document) => (
                <div
                  key={document.id}
                  className="rounded-xl border border-white/8 bg-white/[.025] p-4"
                >
                  <div className="flex items-start gap-3">
                    <div className="grid size-9 shrink-0 place-items-center rounded-lg bg-white/5 text-slate-400">
                      <FileText className="size-4" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium">
                        {document.filename}
                      </p>
                      <p className="mt-1 text-xs text-slate-500">
                        {document.page_count} 页 ·{' '}
                        {document.size_bytes
                          ? `${Math.ceil(document.size_bytes / 1024)} KB`
                          : '大小未知'}
                      </p>
                    </div>
                    <Badge
                      variant="outline"
                      className={
                        document.status === 'indexed'
                          ? 'border-emerald-400/20 text-emerald-300'
                          : document.status === 'failed'
                            ? 'border-rose-400/20 text-rose-300'
                            : 'border-amber-400/20 text-amber-300'
                      }
                    >
                      {statusLabels[document.status] ?? document.status}
                    </Badge>
                  </div>
                  {document.error_message && (
                    <p className="mt-3 text-xs text-rose-300">
                      {document.error_message}
                    </p>
                  )}
                  {(document.status === 'parsed' ||
                    document.status === 'chunked') && (
                    <Button
                      onClick={() => void advanceDocument(document)}
                      disabled={busyDocument === document.id}
                      className="mt-3 w-full bg-white/8 text-slate-100 hover:bg-white/12"
                    >
                      {busyDocument === document.id ? (
                        <LoaderCircle className="animate-spin" />
                      ) : document.status === 'parsed' ? (
                        <Boxes />
                      ) : (
                        <RefreshCw />
                      )}
                      {document.status === 'parsed'
                        ? '生成文本切片'
                        : '建立向量索引'}
                    </Button>
                  )}
                </div>
              ))}
              {!documents.length && (
                <div className="py-10 text-center text-sm text-slate-500">
                  还没有文档
                </div>
              )}
            </div>
          </div>
        </SheetContent>
      </Sheet>

      <Sheet open={sourceOpen} onOpenChange={setSourceOpen}>
        <SheetContent className="w-full border-white/10 bg-[#0d1929] text-slate-100 sm:max-w-2xl">
          <SheetHeader className="border-b border-white/8 p-5">
            <SheetTitle className="text-slate-100">引用资料</SheetTitle>
            <SheetDescription className="text-slate-400">
              回答所依据的知识库原文，可按资料编号核对。
            </SheetDescription>
          </SheetHeader>
          <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
            {selectedSources.map((source, index) => (
              <article
                key={source.chunk_id}
                className="rounded-xl border border-white/8 bg-white/[.025] p-4"
              >
                <div className="mb-3 flex flex-wrap items-center gap-2">
                  <Badge className="bg-cyan-400 text-[#07111f]">
                    {source.citation || `资料${index + 1}`}
                  </Badge>
                  <span className="text-xs text-slate-400">
                    {source.filename}
                  </span>
                  {source.section_title && (
                    <span className="text-xs text-slate-500">
                      · {source.section_title}
                    </span>
                  )}
                  <span className="ml-auto text-xs tabular-nums text-slate-500">
                    相关度 {source.score.toFixed(3)}
                  </span>
                </div>
                <p className="whitespace-pre-wrap text-sm leading-6 text-slate-300">
                  {source.content}
                </p>
              </article>
            ))}
          </div>
        </SheetContent>
      </Sheet>
    </main>
  );
}
