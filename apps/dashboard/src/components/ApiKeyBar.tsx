"use client";

import { useEffect, useState } from "react";
import { apiKey, setApiKey } from "@/lib/api";

export default function ApiKeyBar({ onSaved }: { onSaved?: () => void }) {
  const [value, setValue] = useState("");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setValue(apiKey());
  }, []);

  return (
    <div className="card flex items-center gap-3">
      <span className="stat-label whitespace-nowrap">X-API-Key</span>
      <input
        className="flex-1 rounded-md border border-slate-300 px-3 py-1.5 text-sm"
        placeholder="nf_... (from the API's bootstrap-admin log line, or /api/v1/api-keys)"
        value={value}
        onChange={(e) => setValue(e.target.value)}
      />
      <button
        className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:opacity-90"
        onClick={() => {
          setApiKey(value);
          onSaved?.(); // reload with the new key instead of leaving the old 401 on screen
          setSaved(true);
          setTimeout(() => setSaved(false), 1500);
        }}
      >
        {saved ? "Saved" : "Save"}
      </button>
    </div>
  );
}
