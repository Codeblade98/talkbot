from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from tools import stt, chat, tts, check_conversation_end, pronunciation_assessment
from utils import AgentState

import pyaudio

glob_state = {
    "messages": [],
    "actions": [],
    "action_num": 0,
    "max_speaking_time": 20,
    "audio_params": {
        "sample_rate": 16000,
        "channels": 1,
        "sample_width": 2,
        "format_": pyaudio.paInt16,
        "chunk": 1024
    },
    "token_usage_history": [],
    "total_input_tokens": 0,
    "total_output_tokens": 0
}

checkpoint_config = {
    "thread_id": "user123"
}

def create_graph():
    graph = StateGraph(AgentState)

    graph.add_node("stt", stt)            
    graph.add_node("chat", chat)        
    graph.add_node("tts", tts)   
    graph.add_node("pronunciation_assessment", pronunciation_assessment)      

    graph.set_entry_point("stt")
    graph.add_conditional_edges(
        "tts", check_conversation_end, {"stt": "stt", END: END}
    )
    graph.add_edge("stt", "chat")              
    graph.add_edge("chat", "tts") 
    # graph.add_edge("pronunciation_assessment", "tts")            

    memory = MemorySaver()
    workflow = graph.compile(checkpointer=memory)

    return workflow

def draw_workflow(workflow):
    workflow_image = workflow.get_graph().draw_mermaid_png()
    return workflow_image

def initialize_workflow(workflow, messages=None, actions=None, action_num=None, max_speaking_time=None, audio_params=None, token_usage_history=None, total_input_tokens=None, total_output_tokens=None) -> AgentState:
    updates = {k: v for k, v in locals().items() if v is not None}
    global glob_state

    glob_state.update(updates)

def run_workflow(workflow):
    global glob_state, checkpoint_config
    workflow.invoke(glob_state, checkpoint_config)

def main():
    workflow = create_graph()
    workflow_img = draw_workflow(workflow)
    max_speaking_time = int(input("Enter the maximum speaking time: "))
    initialize_workflow(workflow, max_speaking_time=max_speaking_time)
    run_workflow(workflow)

if __name__ == "__main__":
    main()