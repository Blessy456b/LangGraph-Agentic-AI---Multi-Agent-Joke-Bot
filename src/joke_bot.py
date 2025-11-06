from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

# Initialize the Groq model (make sure GROQ_API_KEY is set)
model = ChatGroq(model="llama-3.1-8b-instant")  # or "mixtral-8x7b" if you prefer

# Define the joke-telling function
def tell_joke(state):
    topic = state.get("topic", "random")
    response = model.invoke([HumanMessage(content=f"Tell me a funny joke about {topic}")])
    return {"joke": response.content, "topic": topic}

# Create the LangGraph workflow
graph = StateGraph(dict)
graph.add_node("tell_joke", tell_joke)
graph.set_entry_point("tell_joke")
graph.add_edge("tell_joke", END)
app = graph.compile()

# Interactive loop
if __name__ == "__main__":
    print("😂 Welcome to the LangGraph Joke Bot (Grok Edition)!")
    print("Type a topic to get a joke, or type 'exit' to quit.\n")

    while True:
        topic = input("Enter a topic: ").strip()
        if topic.lower() in ["exit", "quit"]:
            print("\nThanks for laughing with me! 👋")
            break

        result = app.invoke({"topic": topic})
        print("\nHere's your joke:\n")
        print(result["joke"])
        print("-" * 60)
