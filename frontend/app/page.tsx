"use client";

import { useEffect, useRef, useState } from "react";
import { Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MessageBubble } from "@/components/message-bubble";
import { EXPLAIN_TRIGGER, EXPLAIN_CONTENT, CLEAR_TRIGGER } from "@/lib/explain";

export type Message = {
  role: "user" | "assistant";
  content: string;
  tool?: string;
  isStreaming?: boolean;
};

const EXAMPLES = [
  {
    tool: "vector_search",
    label: "Vector search",
    color: "bg-blue-100 text-blue-800 border-blue-200",
    questions: [
      "What is the maximum out-of-pocket limit for Medicare Advantage plans?",
      "What does §422.112 say about network adequacy standards?",
    ],
  },
  {
    tool: "sql_query",
    label: "SQL",
    color: "bg-emerald-100 text-emerald-800 border-emerald-200",
    questions: [
      "How many contracts received a 5-star overall rating?",
      "Which parent organizations have the most 4-star-or-above contracts?",
    ],
  },
  {
    tool: "openfda_search",
    label: "openFDA",
    color: "bg-orange-100 text-orange-800 border-orange-200",
    questions: [
      "What are the contraindications for Eliquis?",
      "Has Metformin been recalled recently?",
    ],
  },
];

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSlow, setIsSlow] = useState(false);
  const slowTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSlow]);

  function clearSlowTimer() {
    if (slowTimerRef.current) {
      clearTimeout(slowTimerRef.current);
      slowTimerRef.current = null;
    }
    setIsSlow(false);
  }

  async function sendMessage(override?: string) {
    const question = (override ?? input).trim();
    if (!question || isLoading) return;
    setInput("");
    setError(null);

    if (question.toUpperCase() === EXPLAIN_TRIGGER) {
      setMessages((prev) => [
        ...prev,
        { role: "user", content: question },
        { role: "assistant", content: EXPLAIN_CONTENT, isStreaming: false },
      ]);
      return;
    }

    if (question.toUpperCase() === CLEAR_TRIGGER) {
      setMessages([]);
      setError(null);
      return;
    }

    setIsLoading(true);
    clearSlowTimer();
    slowTimerRef.current = setTimeout(() => setIsSlow(true), 10_000);

    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", isStreaming: true },
    ]);

    try {
      const resp = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/query`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
        }
      );

      if (!resp.ok || !resp.body) {
        throw new Error(`Server error: HTTP ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const lines = decoder
          .decode(value)
          .split("\n")
          .filter((l) => l.startsWith("data: "));

        for (const line of lines) {
          const event = JSON.parse(line.slice(6));

          if (event.type === "token") {
            setMessages((prev) => {
              const last = prev[prev.length - 1];
              return [
                ...prev.slice(0, -1),
                { ...last, content: last.content + event.value },
              ];
            });
          } else if (event.type === "tool") {
            setMessages((prev) => {
              const last = prev[prev.length - 1];
              return [...prev.slice(0, -1), { ...last, tool: event.name }];
            });
          } else if (event.type === "done") {
            setMessages((prev) => {
              const last = prev[prev.length - 1];
              return [
                ...prev.slice(0, -1),
                { ...last, isStreaming: false },
              ];
            });
          } else if (event.type === "error") {
            throw new Error(event.message);
          }
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last?.role === "assistant" && last.isStreaming) {
          return prev.slice(0, -1);
        }
        return prev;
      });
    } finally {
      setIsLoading(false);
      clearSlowTimer();
    }
  }

  return (
    <main className="flex flex-col h-dvh w-full">
      <header className="border-b border-neutral-200 shrink-0">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 py-4">
          <h1 className="text-xl font-semibold text-neutral-900">
            Healthcare RAG Agent
          </h1>
          <p className="text-sm text-neutral-500 mt-0.5">
            CMS regulations · Star Ratings · FDA drug data
          </p>
        </div>
      </header>

      <div className="bg-amber-50 border-b border-amber-200 shrink-0">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 py-2 text-xs text-amber-900">
          <strong>Demo only.</strong> AI-generated output may be inaccurate or
          fabricated. <strong>Not medical, legal, or compliance advice.</strong>{" "}
          Do not use for any clinical, regulatory, or professional decision.
          Consult a qualified professional and primary sources.
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 py-4 space-y-4">
          {messages.length === 0 && (
            <div className="py-12 space-y-6">
              <p className="text-center text-neutral-400 text-sm">
                Try one of these, or ask your own question.{" "}
                <button
                  onClick={() => sendMessage(EXPLAIN_TRIGGER)}
                  className="text-neutral-600 underline underline-offset-2 hover:text-neutral-900"
                >
                  Type {EXPLAIN_TRIGGER} for a project tour.
                </button>
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {EXAMPLES.map((group) => (
                  <div key={group.tool} className="space-y-2">
                    <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold ${group.color}`}>
                      {group.label}
                    </span>
                    {group.questions.map((q) => (
                      <button
                        key={q}
                        onClick={() => sendMessage(q)}
                        className="w-full text-left rounded-lg border border-neutral-200 bg-white px-3 py-2.5 text-sm text-neutral-700 hover:border-neutral-400 hover:bg-neutral-50 transition-colors"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}
          {messages.map((msg, i) => (
            <MessageBubble key={i} message={msg} />
          ))}
          {isSlow && isLoading && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
              Sorry, this is taking longer than expected. The free Groq API tier deprioritizes requests, so inference can sometimes exceed 60 seconds.
            </div>
          )}
          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      <footer className="border-t border-neutral-200 shrink-0">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 py-4">
          <div className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) =>
                e.key === "Enter" && !e.shiftKey && sendMessage()
              }
              placeholder="Ask a question..."
              disabled={isLoading}
              className="flex-1"
            />
            <Button
              onClick={() => sendMessage()}
              disabled={isLoading || !input.trim()}
              size="icon"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
          {messages.length > 0 && (
            <p className="text-xs text-neutral-400 text-center pt-1">
              Type <span className="font-mono">CLEAR</span> to reset and see example prompts
            </p>
          )}
        </div>
      </footer>
    </main>
  );
}
