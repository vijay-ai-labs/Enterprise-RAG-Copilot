import React, { useCallback, useEffect, useState } from "react";
import {
  deleteConversation,
  fetchConversations,
  ConversationSummary,
} from "../api/client";
import { MessageSquare, Plus, Trash2 } from "lucide-react";

interface Props {
  currentConversationId: string;
  onSelect: (id: string) => void;
  onNew: () => void;
}

export default function ConversationSidebar({ currentConversationId, onSelect, onNew }: Props) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);

  const load = useCallback(() => {
    fetchConversations(30)
      .then((r) => setConversations(r.conversations))
      .catch(() => {});
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, 15_000);
    return () => clearInterval(id);
  }, [load]);

  const handleDelete = async (e: React.MouseEvent, convId: string) => {
    e.stopPropagation();
    await deleteConversation(convId).catch(() => {});
    setConversations((prev) => prev.filter((c) => c.conversation_id !== convId));
    if (convId === currentConversationId) onNew();
  };

  const formatTime = (ts: number) => {
    const d = new Date(ts * 1000);
    const now = new Date();
    const diffH = (now.getTime() - d.getTime()) / 3_600_000;
    if (diffH < 1) return `${Math.round(diffH * 60)}m ago`;
    if (diffH < 24) return `${Math.round(diffH)}h ago`;
    return d.toLocaleDateString();
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <MessageSquare size={11} className="text-slate-500" />
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Conversations
          </span>
        </div>
        <button
          onClick={onNew}
          title="New conversation"
          className="p-1 rounded text-slate-500 hover:text-brand-400 hover:bg-slate-700/50 transition-colors"
        >
          <Plus size={13} />
        </button>
      </div>

      {conversations.length === 0 ? (
        <p className="text-xs text-slate-600 text-center py-1">No conversations yet</p>
      ) : (
        <ul className="flex flex-col gap-0.5 max-h-48 overflow-y-auto pr-1">
          {conversations.map((c) => {
            const isActive = c.conversation_id === currentConversationId;
            return (
              <li key={c.conversation_id}>
                <button
                  onClick={() => onSelect(c.conversation_id)}
                  className={`w-full text-left px-2 py-1.5 rounded-lg flex items-start gap-2 group transition-colors ${
                    isActive
                      ? "bg-brand-600/20 border border-brand-500/30"
                      : "hover:bg-slate-700/40 border border-transparent"
                  }`}
                >
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-slate-300 truncate leading-tight">
                      {c.preview || "Empty conversation"}
                    </p>
                    <p className="text-xs text-slate-600 mt-0.5">
                      {c.turn_count / 2 | 0} turns · {formatTime(c.last_active)}
                    </p>
                  </div>
                  <button
                    onClick={(e) => handleDelete(e, c.conversation_id)}
                    title="Delete"
                    className="shrink-0 opacity-0 group-hover:opacity-100 p-0.5 text-slate-600 hover:text-rose-400 transition-all"
                  >
                    <Trash2 size={11} />
                  </button>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
