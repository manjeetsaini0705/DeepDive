from agents import build_reader_agent, build_search_agent,critic_chain,writer_chain

def run_research_pipeline(topic:str)->dict:
    state={}
    
    #_________________search agent working
    print("\n"+" ="*50)
    print("step 1- search agent is working......")
    print("="*50)
    
    searhc_agent = build_search_agent()
    search_result = searhc_agent.invoke({
             "messages":[("user", f"find recent, reliable and detailed information about:{topic}")]   

    })
    
    search_messages = search_result["messages"]

    tool_messages = [
        msg for msg in search_messages
        if msg.__class__.__name__ == "ToolMessage"
    ]

    if tool_messages:
        state["search_results"] = "\n\n".join(
            str(msg.content) for msg in tool_messages
        )
    else:
        state["search_results"] = search_messages[-1].content
    
    print("\n search result",state["search_results"])
    
    
    
    
    #_________________reader agent working
    print("\n"+" ="*50)
    print("step 2-Reader agent is scraping top resources......")
    print("="*50)
    
    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"""You are given search results for the topic:

{topic}

Search Results:
{state["search_results"]}

Identify the most relevant URL from the search results.

Then use the scrape_url tool with that URL.

Do NOT simply summarize the search results.
Actually call scrape_url on one of the URLs.
Return the scraped content."""
            )
        ]
    })

    state["scraped_content"] = reader_result["messages"][-1].content

    print("\nSCRAPED CONTENT:\n")
    print(state["scraped_content"])


    
    
    
    
    #_________________writer chain
    print("\n"+" ="*50)
    print("step 3-Writer is drafting the report......")
    print("="*50)
    
    research_combined = (
        f"SEARCH RESULTS: \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT: \n {state['scraped_content']}"
    )
    
    state["report"] = writer_chain.invoke({
        "topic":topic,
        "research":research_combined
    })
    
    print("\n Final Report\n", state['report'])
    
    
    
    
    
    #_________________critic chain
    print("\n"+" ="*50)
    print("step 4-Critic is reviewing the report......")
    print("="*50)
    
    state["feedback"]= critic_chain.invoke({
        "report":state['report']
    })
    print("\n critic report \n",state['feedback'])
    
    return state
    
    
if __name__ == "__main__":
    topic = input("\n Enter a research topic: ")
    run_research_pipeline(topic)
