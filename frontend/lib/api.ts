export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000/api/v1';

export type KnowledgeBase = {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
};

export type DocumentItem = {
  id: string;
  filename: string;
  status: string;
  size_bytes: number | null;
  page_count: number;
  error_message: string | null;
};

export type Conversation = {
  id: string;
  knowledge_base_id: string | null;
  title: string;
  created_at: string;
  updated_at: string;
};

export type Source = {
  citation?: string;
  chunk_id: string;
  document_id: string;
  filename: string;
  content: string;
  score: number;
  page_number: number | null;
  section_title: string | null;
  retrieval_sources?: string[];
  rerank_score?: number | null;
};

export type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations: Source[];
  created_at?: string;
  streaming?: boolean;
  progress?: string;
};

type Envelope<T> = { code: number; message: string; data: T };

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  const payload = (await response.json()) as Envelope<T>;
  if (!response.ok || payload.code !== 0) {
    throw new Error(payload.message || `请求失败（${response.status}）`);
  }
  return payload.data;
}

export async function parseSse(
  response: Response,
  onEvent: (event: string, data: Record<string, unknown>) => void,
) {
  if (!response.ok || !response.body) {
    const payload: unknown = await response.json().catch(() => null);
    const message =
      typeof payload === 'object' &&
      payload !== null &&
      'message' in payload &&
      typeof payload.message === 'string'
        ? payload.message
        : `请求失败（${response.status}）`;
    throw new Error(message);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, '\n');
    const blocks = buffer.split('\n\n');
    buffer = blocks.pop() ?? '';
    for (const block of blocks) {
      let event = 'message';
      const dataLines: string[] = [];
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
      }
      if (dataLines.length) onEvent(event, JSON.parse(dataLines.join('\n')));
    }
    if (done) break;
  }
}
