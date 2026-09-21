"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import LineageGraph from "@/components/LineageGraph";
import { apiGet, type LineageGraph as LineageGraphData } from "@/lib/api";

export default function GenomeLineagePage() {
  const params = useParams<{ systemId: string }>();
  const [graph, setGraph] = useState<LineageGraphData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiGet<LineageGraphData>(`/api/v1/genomes/${params.systemId}/lineage`)
      .then(setGraph)
      .catch((err) => setError((err as Error).message));
  }, [params.systemId]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Evolution Graph — {params.systemId}</h1>
        <p className="text-sm text-slate-500">
          The baseline and each experiment&rsquo;s selected winner for this application, with the
          mutation lineage connecting them. Node color is the genome&rsquo;s current promotion
          status; click a node to inspect it.
        </p>
      </div>
      {error && <div className="card text-sm text-bad">{error}</div>}
      {graph && <LineageGraph graph={graph} />}
    </div>
  );
}
