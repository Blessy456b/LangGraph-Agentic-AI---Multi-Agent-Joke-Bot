# multi_agent_joke_bot.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq  # or replace with another chat model (e.g., ChatOpenAI, ChatGoogleGenerativeAI)

# Use Groq model (you can change model name)
llm = ChatGroq(model="llama-3.1-8b-instant")

# Define the graph state structure
class JokeState(dict):
    topic: str
    joke: str
    critique: str
    rewritten: str

# --- NODE 1: JOKE GENERATOR ---
def generate_joke(state: JokeState):
    topic = state.get("topic", "random stuff")
    prompt = f"Tell a funny and short joke about {topic}. Return only the joke."
    response = llm.invoke([HumanMessage(content=prompt)])
    joke_text = response.content.strip() if hasattr(response, "content") else str(response)
    return {"joke": joke_text or "No joke generated."}

# --- NODE 2: CRITIC ---
def critique_joke(state: JokeState):
    joke = state.get("joke", "")
    if not joke:
        return {"critique": "No joke to critique."}
    prompt = f"Critique this joke for humor and clarity:\n\n{joke}\n\nReturn only the feedback in one sentence."
    response = llm.invoke([HumanMessage(content=prompt)])
    critique_text = response.content.strip() if hasattr(response, "content") else str(response)
    return {"critique": critique_text or "No critique."}

# --- NODE 3: REWRITER ---
def rewrite_joke(state: JokeState):
    joke = state.get("joke", "")
    if not joke:
        return {"rewritten": "No joke to rewrite."}
    prompt = f"Rewrite this joke to make it family-friendly and clean while keeping it funny:\n\n{joke}"
    response = llm.invoke([HumanMessage(content=prompt)])
    rewritten_text = response.content.strip() if hasattr(response, "content") else str(response)
    return {"rewritten": rewritten_text or "No rewritten version."}

# --- BUILD THE GRAPH ---
graph = StateGraph(JokeState)

graph.add_node("joke_gen", generate_joke)
graph.add_node("critic", critique_joke)
graph.add_node("rewriter", rewrite_joke)

graph.set_entry_point("joke_gen")
graph.add_edge("joke_gen", "critic")
graph.add_edge("critic", "rewriter")
graph.add_edge("rewriter", END)

compiled = graph.compile()

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("😂 Welcome to the Multi-Agent LangGraph Joke Bot!")
    print("Type a topic for a joke or 'exit' to quit.\n")

    while True:
        topic = input("Enter a topic: ").strip()
        if topic.lower() in ["exit", "quit"]:
            print("👋 Goodbye! Keep laughing!")
            break

        inputs = {"topic": topic}
        result = compiled.invoke(inputs)

        print("\n🧠 Original Joke:\n", result.get("joke", "❌ No joke generated."))
        print("\n🤔 Critic’s Review:\n", result.get("critique", "❌ No critique."))
        print("\n😇 Family-Friendly Version:\n", result.get("rewritten", "❌ No rewritten version."))
        print("------------------------------------------------------------")
