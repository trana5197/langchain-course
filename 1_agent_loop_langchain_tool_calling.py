from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage

from langchain.tools import tool

from langsmith import traceable


MODEL = "qwen3:14b"
# MODEL = "gpt-5.5"
MAX_ITERATIONS = 10

# --- Tools (LangChain @tool decorator) ---

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# ----- Agent Loop ------

@traceable(name="agent_under_the_hood_loop_demo")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tool_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    # llm = init_chat_model(f"openai:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Thought process of llm and result is stored in ai_message
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            # answer if not tool_calls in ai_message
            print(f"\nFinal Answer: {ai_message.content}\n")
            return ai_message.content
        
        # if tool_calls in ai_message, this is where we extract values from tool_calls
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}") 
        tool_to_call = tool_dict.get(tool_name)

        if tool_to_call is None:
            raise ValueError(f"Error: Tool '{tool_name}' not found.")

        #  invoking tool call in agent or application and storing result in observation variable  
        observation = tool_to_call.invoke(tool_args)
        print(f"Tool Result: {observation}")

        # sending ai_message and tool observation back to llm in next iteration through messages list
        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_id))

    print("Error: Maximum iterations reached without a final answer.")
    return None



if __name__ == "__main__":
    print("Welcome to the Agent Under The Hood Loop Demo!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")