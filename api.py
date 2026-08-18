"""
api.py: FastAPI production backend with Lazy Imports and LangGraph integration.
"""
import os
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ESL AI Tutor Engine API")

class TutorRequest(BaseModel):
    user_input: str
    proficiency_level: str = "A2"
    current_slide: int = 1
    openai_api_key: str

# Lazy Global Variables
_graph_app = None

def get_langgraph_app(api_key: str):
    """Lazy imports LangChain and LangGraph dependencies on first call."""
    global _graph_app
    if _graph_app is None:
        from typing import TypedDict
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langgraph.graph import StateGraph, END

        class AgentState(TypedDict):
            user_input: str
            level: str
            slide: int
            rag_context: str
            response: str

        def retrieve_pedagogy(state: AgentState):
            from model import initialize_vector_db
            vdb = initialize_vector_db(openai_api_key=api_key)
            docs = vdb.similarity_search(f"Scaffolding rules for {state['level']} level", k=1)
            context = docs[0].page_content if docs else "Keep language simple and encouraging."
            return {"rag_context": context}

        def generate_response(state: AgentState):
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.3)
            prompt = ChatPromptTemplate.from_template(
                "You are an ESL Tutor for Slide #{slide}.\n"
                "Target Level: {level}\n"
                "Pedagogical Guideline: {rag_context}\n"
                "Student said: {user_input}\n"
                "Respond by acknowledging their attempt, gently correcting error if any, and asking a follow-up target question."
            )
            chain = prompt | llm
            res = chain.invoke(state)
            return {"response": res.content}

        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve", retrieve_pedagogy)
        workflow.add_node("generate", generate_response)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        _graph_app = workflow.compile()
        
    return _graph_app

@app.get("/")
def health_check():
    return {"status": "ok", "service": "ESL AI Tutor API"}

@app.post("/v1/tutor/respond")
async def respond(req: TutorRequest):
    try:
        graph = get_langgraph_app(req.openai_api_key)
        initial_state = {
            "user_input": req.user_input,
            "level": req.proficiency_level,
            "slide": req.current_slide,
            "rag_context": "",
            "response": ""
        }
        final_state = await graph.ainvoke(initial_state)
        return {"output": final_state["response"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))