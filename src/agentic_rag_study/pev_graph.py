from __future__ import annotations

import operator
import re
from typing import Annotated, Any, TypedDict

from .retrieval import HybridRetriever

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional dependency fallback
    END = START = StateGraph = None


class PlanStep(TypedDict):
    id: int
    sub_query: str
    tool: str
    depends_on: list[int]


class PEVState(TypedDict, total=False):
    query: str
    query_type: str
    plan: list[PlanStep]
    evidence: Annotated[list[dict[str, Any]], operator.add]
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    verification_result: str
    verification_feedback: str
    final_answer: str
    iteration_count: int
    total_tool_calls: int
    trace: Annotated[list[dict[str, Any]], operator.add]


class HeuristicPlanner:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.aliases = self._collect_aliases()

    def _collect_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for chunk in self.retriever.chunks:
            metadata_aliases = chunk.metadata.get("aliases", [])
            for alias in metadata_aliases:
                aliases[str(alias)] = str(alias)
            if chunk.company:
                aliases[chunk.company] = chunk.company
        return dict(sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True))

    def extract_entities(self, query: str) -> list[str]:
        entities: list[str] = []
        for alias in self.aliases:
            if alias in query and alias not in entities:
                entities.append(alias)
        return entities

    def extract_metric(self, query: str) -> str:
        for metric in ("净利润", "营业收入", "门店数"):
            if metric in query:
                return metric
        return "营业收入"

    def is_comparison(self, query: str) -> bool:
        markers = ("哪家", "谁", "更高", "更低", "比较")
        return any(marker in query for marker in markers)

    def plan(self, state: PEVState) -> list[PlanStep]:
        query = state["query"]
        metric = self.extract_metric(query)
        entities = self.extract_entities(query)
        tool = "keyword_search" if "精确匹配" in state.get("verification_feedback", "") else "hybrid_search"

        if self.is_comparison(query) and len(entities) >= 2:
            return [
                {"id": 1, "sub_query": f"{entities[0]} {metric}", "tool": tool, "depends_on": []},
                {"id": 2, "sub_query": f"{entities[1]} {metric}", "tool": tool, "depends_on": []},
            ]

        if entities:
            return [
                {"id": 1, "sub_query": entities[0], "tool": tool, "depends_on": []},
                {"id": 2, "sub_query": metric, "tool": tool, "depends_on": [1]},
            ]

        return [{"id": 1, "sub_query": query, "tool": tool, "depends_on": []}]


def _extract_company_from_results(results: list[dict[str, Any]]) -> str:
    for result in results:
        company = result.get("metadata", {}).get("company", "")
        if company:
            return company
    return ""


def _extract_metric_value(text: str, metric: str) -> str:
    patterns = [
        rf"{re.escape(metric)}[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*(亿元|家|个)?",
        r"([0-9]+(?:\.[0-9]+)?)\s*(亿元|家|个)",
    ]
    for pattern in patterns:
        matched = re.search(pattern, text)
        if not matched:
            continue
        number = matched.group(1)
        unit = matched.group(2) if len(matched.groups()) >= 2 else ""
        return f"{number}{unit or ''}"
    return ""


def planner_node(planner: HeuristicPlanner):
    def _node(state: PEVState) -> PEVState:
        plan = planner.plan(state)
        next_iteration = state.get("iteration_count", 0) + 1
        return {
            "plan": plan,
            "iteration_count": next_iteration,
            "trace": [{"node": "planner", "plan": plan, "iteration": next_iteration}],
        }

    return _node


def executor_node(retriever: HybridRetriever):
    def _node(state: PEVState) -> PEVState:
        evidence_lookup = {item["step_id"]: item for item in state.get("evidence", [])}
        new_evidence: list[dict[str, Any]] = []
        new_calls: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []

        for step in state["plan"]:
            query_parts: list[str] = []
            for dependency in step["depends_on"]:
                previous = evidence_lookup.get(dependency)
                if previous:
                    company = _extract_company_from_results(previous["results"])
                    if company:
                        query_parts.append(company)
            query_parts.append(step["sub_query"])
            final_query = " ".join(part for part in query_parts if part).strip()
            results = retriever.dispatch(step["tool"], final_query, top_k=3)
            record = {
                "step_id": step["id"],
                "query": final_query,
                "tool": step["tool"],
                "results": [item.to_record() for item in results],
            }
            evidence_lookup[step["id"]] = record
            new_evidence.append(record)
            new_calls.append({"step_id": step["id"], "tool": step["tool"], "query": final_query})
            trace.append({"node": "executor", "step_id": step["id"], "query": final_query, "hits": len(results)})

        return {
            "evidence": new_evidence,
            "tool_calls": new_calls,
            "total_tool_calls": state.get("total_tool_calls", 0) + len(new_calls),
            "trace": trace,
        }

    return _node


def verifier_node():
    def _node(state: PEVState) -> PEVState:
        latest_plan_ids = {step["id"] for step in state["plan"]}
        latest_evidence = [item for item in state.get("evidence", []) if item["step_id"] in latest_plan_ids]
        missing = [item["step_id"] for item in latest_evidence if not item["results"]]
        if missing:
            return {
                "verification_result": "insufficient",
                "verification_feedback": "缺少关键证据，请改用精确匹配并缩短查询词。",
                "trace": [{"node": "verifier", "result": "insufficient", "missing_steps": missing}],
            }

        if len(latest_evidence) < len(state["plan"]):
            return {
                "verification_result": "insufficient",
                "verification_feedback": "执行步骤不足，请补齐剩余步骤。",
                "trace": [{"node": "verifier", "result": "insufficient", "missing_steps": "incomplete"}],
            }

        return {
            "verification_result": "sufficient",
            "verification_feedback": "",
            "trace": [{"node": "verifier", "result": "sufficient"}],
        }

    return _node


def synthesizer_node(planner: HeuristicPlanner):
    def _node(state: PEVState) -> PEVState:
        query = state["query"]
        metric = planner.extract_metric(query)
        evidence = [item for item in state.get("evidence", []) if item["results"]]
        if not evidence:
            return {"final_answer": "未找到足够证据", "trace": [{"node": "synthesizer", "answer": "未找到足够证据"}]}

        if planner.is_comparison(query):
            comparisons: list[tuple[str, float]] = []
            for item in evidence:
                if metric not in item["query"]:
                    continue
                company = _extract_company_from_results(item["results"])
                value = _extract_metric_value(item["results"][0]["text"], metric)
                if not company or not value:
                    continue
                number = float(re.findall(r"[0-9]+(?:\.[0-9]+)?", value)[0])
                comparisons.append((company, number))
            if comparisons:
                reverse = "更低" not in query
                winner = sorted(comparisons, key=lambda pair: pair[1], reverse=reverse)[0][0]
                return {"final_answer": winner, "trace": [{"node": "synthesizer", "answer": winner}]}

        latest = evidence[-1]
        top_result = latest["results"][0]
        answer = _extract_metric_value(top_result["text"], metric)
        if not answer:
            answer = top_result.get("metadata", {}).get("company", "") or top_result["text"][:40]
        return {"final_answer": answer, "trace": [{"node": "synthesizer", "answer": answer}]}

    return _node


def route_after_verifier(state: PEVState) -> str:
    if state.get("verification_result") == "sufficient":
        return "synthesizer"
    if state.get("iteration_count", 0) >= 2:
        return "synthesizer"
    return "planner"


def build_pev_graph(retriever: HybridRetriever):
    if StateGraph is None:
        raise ImportError("langgraph 未安装，无法构建 PEV 图。请先安装 requirements.txt。")

    planner = HeuristicPlanner(retriever)
    graph = StateGraph(PEVState)
    graph.add_node("planner", planner_node(planner))
    graph.add_node("executor", executor_node(retriever))
    graph.add_node("verifier", verifier_node())
    graph.add_node("synthesizer", synthesizer_node(planner))
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "verifier")
    graph.add_conditional_edges("verifier", route_after_verifier, {"planner": "planner", "synthesizer": "synthesizer"})
    graph.add_edge("synthesizer", END)
    return graph.compile()
