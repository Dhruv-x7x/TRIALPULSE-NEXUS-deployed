
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
from src.knowledge.rag_knowledge_base import RAGKnowledgeBase, RetrievalChain
from src.agents.llm_wrapper import get_llm

logger = logging.getLogger(__name__)

class GenerativeDocumentEngine:
    """
    RAG-powered engine for generating context-aware clinical trial reports.
    Uses ICH-GCP guidelines, SOPs, and Protocol knowledge.
    """
    
    def __init__(self):
        self.llm = get_llm()
        self.kb = RAGKnowledgeBase()
        self.retriever = None
        
        # Initialize RAG if index exists
        if self.kb.load_index():
            self.retriever = RetrievalChain(self.kb)
            logger.info("RAG Knowledge Base linked to Document Engine")
        else:
            logger.warning("RAG Index not found. Using zero-shot generation.")

    def generate_report(self, report_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a structured narrative using RAG context."""
        
        query = f"Provide a {report_type} summary for clinical trial data: {context.get('issues', '')}"
        
        rag_context = ""
        if self.retriever:
            # Get relevant ICH-GCP or SOP context
            rag_context = self.kb.get_relevant_context(query, max_tokens=1000)
            
        system_prompt = f"""You are the TrialPulse Nexus Generative Document Engine.
Your goal is to produce high-quality, professional clinical trial narrative summaries.
Use the provided RAG context (ICH-GCP guidelines and SOPs) to ensure regulatory compliance.

RAG CONTEXT:
{rag_context}
"""
        
        prompt = f"""Generate a {report_type.replace('_', ' ')} for the following context:
{str(context)}

The response should be a professional 2-3 paragraph summary focusing on data quality, safety, and operational readiness.
DO NOT include technical jargon from the prompt, only the final narrative."""

        llm_res = self.llm.generate(prompt=prompt, system_prompt=system_prompt)
        
        return {
            "report_id": f"REP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "type": report_type,
            "generated_at": datetime.utcnow().isoformat(),
            "content": llm_res.content,
            "metadata": {
                "rag_active": self.retriever is not None,
                "model": llm_res.model,
                "latency_ms": llm_res.latency_ms
            }
        }

    def generate_safety_narrative(self, patient_data: Dict[str, Any], sae_details: Dict[str, Any]) -> str:
        """Specific helper for safety narratives."""
        return self.generate_report("safety_narrative", {**patient_data, **sae_details})["content"]
