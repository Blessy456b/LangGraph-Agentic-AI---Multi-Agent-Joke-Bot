from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import Dict, Any, List

# ---------------------------
# Global Memory Store
# ---------------------------
memory_store: List[str] = []


# ---------------------------
# Define Nodes
# ---------------------------

def tell_joke(state: Dict[str, Any]):
    """Generate a joke for the given topic using Groq model."""
    topic = state["topic"]
    model = ChatGroq(model="llama-3.1-8b-instant")

    # Avoid repeating jokes
    if "previous_jokes" not in state:
        state["previous_jokes"] = memory_store

    # Generate new joke
    response = model.invoke([
        HumanMessage(content=f"Tell me a creative and funny joke about {topic}. Keep it short and witty.")
    ])
    joke = response.content.strip()

    # Check memory
    if joke in memory_store:
        joke = f"(⚠️ Reused joke avoided) Here's a new twist:\n{joke} (again!)"
    else:
        memory_store.append(joke)

    state["joke"] = joke
    return state


def critic_node(state: Dict[str, Any]):
    """Critique the joke and assign a humor score (0–10)."""
    joke = state.get("joke", "")
    model = ChatGroq(model="llama-3.1-8b-instant")

    critique_prompt = f"""Rate the following joke for humor on a scale of 0 to 10, 
then briefly explain your reasoning. 
Joke: {joke}"""
    response = model.invoke([HumanMessage(content=critique_prompt)])

    content = response.content.strip()

    # Extract humor score
    import re
    match = re.search(r"(\b\d{1,2}\b)", content)
    humor_score = int(match.group(1)) if match else 5  # default 5 if no score found

    state["critique"] = content
    state["humor_score"] = humor_score
    return state


def rewriter_node(state: Dict[str, Any]):
    """Rewrite the joke to make it family-friendly if humor score < 7."""
    joke = state.get("joke", "")
    humor_score = state.get("humor_score", 5)
    model = ChatGroq(model="llama-3.1-8b-instant")

    if humor_score >= 7:
        rewritten = "(✅ Already family-friendly and funny enough!)"
    else:
        prompt = f"Rewrite this joke to be funnier and family-friendly:\n{joke}"
        response = model.invoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip()

    state["rewritten"] = rewritten
    return state


def memory_node(state: Dict[str, Any]):
    """Display joke history."""
    state["memory"] = memory_store
    return state


# ---------------------------
# Graph Definition
# ---------------------------
workflow = StateGraph(Dict[str, Any])

workflow.add_node("tell_joke", tell_joke)
workflow.add_node("critic", critic_node)
workflow.add_node("rewriter", rewriter_node)
workflow.add_node("memory", memory_node)

workflow.add_edge("tell_joke", "critic")
workflow.add_edge("critic", "rewriter")
workflow.add_edge("rewriter", "memory")
workflow.add_edge("memory", END)

workflow.set_entry_point("tell_joke")
app = workflow.compile()


# ---------------------------
# Interactive CLI
# ---------------------------
if __name__ == "__main__":
    print("😂 Welcome to the Multi-Agent LangGraph Joke Bot (with Memory + HumorScore)!")
    print("Type a topic for a joke or 'exit' to quit.\n")

    while True:
        topic = input("Enter a topic: ").strip()
        if topic.lower() == "exit":
            break

        result = app.invoke({"topic": topic})

        print("\n🧠 Original Joke:\n", result.get("joke", "❌ No joke generated."))
        print("\n😂 Humor Score:", result.get("humor_score", "❌"))
        print("\n🤔 Critic’s Review:\n", result.get("critique", "❌ No critique."))
        print("\n😇 Family-Friendly Version:\n", result.get("rewritten", "❌ No rewritten version."))
        print("\n🧩 Joke Memory (so far):")
        for i, j in enumerate(result.get("memory", []), 1):
            print(f"   {i}. {j}")
        print("-" * 60)
