"use client";

import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { type Message } from "@/app/page";

const TOOL_LABELS: Record<string, string> = {
  vector_search: "Vector search",
  sql_query: "SQL",
  openfda_search: "openFDA",
};

const TOOL_COLORS: Record<string, string> = {
  vector_search: "bg-blue-100 text-blue-800 border-blue-200",
  sql_query: "bg-emerald-100 text-emerald-800 border-emerald-200",
  openfda_search: "bg-orange-100 text-orange-800 border-orange-200",
};

function CitationChip({ page }: { page: string }) {
  return (
    <Badge
      variant="outline"
      className="mx-0.5 cursor-default font-mono text-xs align-middle"
    >
      p.&nbsp;{page}
    </Badge>
  );
}

function MessageContent({ text }: { text: string }) {
  const parts = text.split(/(\[p\.\s*\d+(?:,\s*p\.\s*\d+)*\])/g);
  return (
    <>
      {parts.map((part, i) => {
        if (/^\[p\./.test(part)) {
          const pages = [...part.matchAll(/\d+/g)].map((m) => m[0]);
          return (
            <span key={i} className="inline-flex flex-wrap gap-0.5 align-middle">
              {pages.map((page) => (
                <CitationChip key={page} page={page} />
              ))}
            </span>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

export function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-neutral-900 px-4 py-2.5 text-white">
          <p className="text-sm leading-relaxed whitespace-pre-wrap">
            {message.content}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] space-y-1.5">
        {message.tool && (
          <Badge
            variant="outline"
            className={TOOL_COLORS[message.tool] ?? ""}
          >
            {TOOL_LABELS[message.tool] ?? message.tool}
          </Badge>
        )}
        <div className="rounded-2xl rounded-tl-sm bg-neutral-100 px-4 py-2.5">
          {message.isStreaming && !message.content ? (
            <div className="space-y-2 py-1">
              <Skeleton className="h-3 w-52" />
              <Skeleton className="h-3 w-64" />
              <Skeleton className="h-3 w-40" />
            </div>
          ) : (
            <div className="text-sm leading-relaxed whitespace-pre-wrap text-neutral-800">
              <MessageContent text={message.content} />
              {message.isStreaming && (
                <span className="animate-pulse ml-0.5 text-neutral-400">▋</span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
