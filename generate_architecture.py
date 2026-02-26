"""Run once to produce static/architecture.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#f8fafc")

# ── colour palette ──────────────────────────────────────────────────────────
C_USER   = "#3b82f6"   # blue
C_API    = "#7c3aed"   # purple
C_AGENT  = "#059669"   # green
C_TOOL   = "#d97706"   # amber
C_DB     = "#dc2626"   # red
C_VEC    = "#0891b2"   # cyan
C_LLM    = "#6d28d9"   # indigo
WHITE    = "white"

def box(ax, x, y, w, h, label, sublabel="", color=C_AGENT, fontsize=10):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.08",
                          linewidth=1.5,
                          edgecolor=color,
                          facecolor=WHITE)
    ax.add_patch(rect)
    rect2 = FancyBboxPatch((x, y + h - 0.36), w, 0.36,
                           boxstyle="round,pad=0.0",
                           linewidth=0,
                           edgecolor=color,
                           facecolor=color)
    ax.add_patch(rect2)
    ax.text(x + w/2, y + h - 0.18, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=WHITE)
    if sublabel:
        ax.text(x + w/2, y + (h - 0.36)/2, sublabel,
                ha="center", va="center", fontsize=8,
                color="#374151", wrap=True,
                multialignment="center")

def arrow(ax, x1, y1, x2, y2, label="", color="#6b7280"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, connectionstyle="arc3,rad=0.0"))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.05, my, label, fontsize=7.5, color=color,
                ha="left", va="center")

# ── title ───────────────────────────────────────────────────────────────────
ax.text(7, 8.65, "Agri-Advisor Pro — System Architecture",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#111827")

# ── User / Client ────────────────────────────────────────────────────────────
box(ax, 0.3, 6.8, 2.2, 1.2, "Client / Browser", "Web UI\n(index.html)", C_USER, 9)

# ── FastAPI layer ────────────────────────────────────────────────────────────
box(ax, 0.3, 4.6, 2.2, 1.7, "FastAPI Server",
    "POST /api/execute\nGET /api/team_info\nGET /api/agent_info\nGET /api/model_architecture",
    C_API, 9)

# ── AgentExecutor ────────────────────────────────────────────────────────────
box(ax, 3.5, 4.6, 3.0, 1.7, "AgentExecutor",
    "LangChain Classic\ncreate_openai_tools_agent\nSteps logger / Callback",
    C_AGENT, 9)

# ── LLM ─────────────────────────────────────────────────────────────────────
box(ax, 3.5, 7.1, 3.0, 1.1, "AgentLLM",
    "LLMod.ai (gpt-5-mini)\nvia OpenAI-compat. API",
    C_LLM, 9)

# ── Tools ────────────────────────────────────────────────────────────────────
box(ax, 7.5, 6.0, 2.8, 1.3, "WeatherTool",
    "16 Israeli cities\nDate-aware lookup\nJSON city_data/",
    C_TOOL, 9)

box(ax, 7.5, 4.2, 2.8, 1.5, "AgriKnowledgeBase",
    "MultiQueryRetriever\nretriever_llm (temp=0)\nRAG over PDF manuals",
    C_TOOL, 9)

# ── Databases ────────────────────────────────────────────────────────────────
box(ax, 11.2, 6.0, 2.4, 1.3, "Supabase\n(PostgreSQL)",
    "Chat sessions\nMessage history",
    C_DB, 9)

box(ax, 11.2, 4.2, 2.4, 1.5, "Pinecone\n(Vector DB)",
    "PDF chunk embeddings\nBAAI/bge-small-en-v1.5",
    C_VEC, 9)

# ── Arrows ───────────────────────────────────────────────────────────────────
# client → fastapi
arrow(ax, 1.4, 6.8, 1.4, 6.3)
# fastapi → agentexecutor
arrow(ax, 2.5, 5.45, 3.5, 5.45)
# agentexecutor ↔ LLM
arrow(ax, 5.0, 6.3, 5.0, 7.1)
arrow(ax, 4.8, 7.1, 4.8, 6.3)
# agentexecutor → weathertool
arrow(ax, 6.5, 5.6, 7.5, 6.4, "tool call")
# agentexecutor → ragTool
arrow(ax, 6.5, 5.1, 7.5, 4.9, "tool call")
# weathertool → supabase (results stored via fastapi)
arrow(ax, 10.3, 6.5, 11.2, 6.5, "read")
# ragTool → pinecone
arrow(ax, 10.3, 4.9, 11.2, 4.9, "query")

# ── legend ───────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_USER,  label="Client"),
    mpatches.Patch(color=C_API,   label="API Layer"),
    mpatches.Patch(color=C_AGENT, label="Agent Core"),
    mpatches.Patch(color=C_LLM,   label="LLM"),
    mpatches.Patch(color=C_TOOL,  label="Tools"),
    mpatches.Patch(color=C_DB,    label="Primary DB"),
    mpatches.Patch(color=C_VEC,   label="Vector DB"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=8,
          framealpha=0.9, ncol=4, bbox_to_anchor=(0.0, 0.0))

plt.tight_layout()
plt.savefig("static/architecture.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved static/architecture.png")
