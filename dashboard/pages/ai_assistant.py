"""
TRIALPULSE NEXUS 10X - AI Assistant Page
Phase 7.2: Fully functional AI-powered natural language interface
Connected to the 6-Agent Orchestration System
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.config.theme import THEME_CONFIG


def get_orchestrator():
    """Get or create the agent orchestrator instance."""
    if 'ai_orchestrator' not in st.session_state:
        try:
            from src.agents.orchestrator import AgentOrchestrator, get_llm_error
            st.session_state.ai_orchestrator = AgentOrchestrator()
            st.session_state.ai_orchestrator_error = None
        except Exception as e:
            st.session_state.ai_orchestrator = None
            # Get more specific error message
            try:
                from src.agents.orchestrator import get_llm_error
                llm_error = get_llm_error()
                if llm_error:
                    st.session_state.ai_orchestrator_error = f"LLM Error: {llm_error}"
                else:
                    st.session_state.ai_orchestrator_error = str(e)
            except:
                st.session_state.ai_orchestrator_error = str(e)
    return st.session_state.ai_orchestrator


def initialize_chat_state():
    """Initialize session state for chat."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = None
    if 'ai_processing' not in st.session_state:
        st.session_state.ai_processing = False


def format_agent_response(state) -> Dict[str, Any]:
    """Format the agent state into displayable sections."""
    sections = {
        'summary': '',
        'key_findings': [],
        'hypotheses': [],
        'recommendations': [],
        'forecasts': [],
        'communications': [],
        'action_plan': '',
        'errors': []
    }
    
    # Get final response
    if hasattr(state, 'final_response') and state.final_response:
        sections['summary'] = state.final_response
    elif hasattr(state, 'summary') and state.summary:
        sections['summary'] = state.summary
    
    # Get action plan
    if hasattr(state, 'action_plan') and state.action_plan:
        sections['action_plan'] = state.action_plan
    
    # Get hypotheses
    if hasattr(state, 'hypotheses'):
        for h in state.hypotheses:
            sections['hypotheses'].append({
                'id': getattr(h, 'hypothesis_id', 'Unknown'),
                'description': getattr(h, 'description', ''),
                'confidence': getattr(h, 'confidence', 0),
                'verification_steps': getattr(h, 'verification_steps', [])
            })
    
    # Get recommendations
    if hasattr(state, 'recommendations'):
        for r in state.recommendations:
            sections['recommendations'].append({
                'id': getattr(r, 'recommendation_id', 'Unknown'),
                'action': getattr(r, 'action', ''),
                'priority': getattr(r, 'priority', None),
                'confidence': getattr(r, 'confidence', 0),
                'impact': getattr(r, 'impact', ''),
                'effort_hours': getattr(r, 'effort_hours', 0),
                'responsible_role': getattr(r, 'responsible_role', ''),
                'requires_approval': getattr(r, 'requires_approval', False)
            })
    
    # Get forecasts
    if hasattr(state, 'forecasts'):
        for f in state.forecasts:
            sections['forecasts'].append({
                'metric': getattr(f, 'metric', ''),
                'prediction': getattr(f, 'prediction', 0),
                'lower_bound': getattr(f, 'lower_bound', 0),
                'upper_bound': getattr(f, 'upper_bound', 0),
                'confidence': getattr(f, 'confidence', 0),
                'timeframe': getattr(f, 'timeframe', ''),
                'assumptions': getattr(f, 'assumptions', []),
                'risks': getattr(f, 'risks', [])
            })
    
    # Get communications
    if hasattr(state, 'communications'):
        for c in state.communications:
            sections['communications'].append({
                'id': getattr(c, 'communication_id', 'Unknown'),
                'type': getattr(c, 'type', 'email'),
                'recipient': getattr(c, 'recipient', ''),
                'subject': getattr(c, 'subject', ''),
                'body': getattr(c, 'body', ''),
                'requires_approval': getattr(c, 'requires_approval', True)
            })
    
    # Get errors
    if hasattr(state, 'errors') and state.errors:
        sections['errors'] = state.errors
    
    return sections


def process_query(query: str) -> Dict[str, Any]:
    """Process a user query through the agent orchestrator."""
    orchestrator = get_orchestrator()
    
    if orchestrator is None:
        return {
            'success': False,
            'error': st.session_state.get('ai_orchestrator_error', 'Orchestrator not initialized'),
            'sections': None
        }
    
    try:
        # Run the orchestrator
        state = orchestrator.run(query)
        
        # Format the response
        sections = format_agent_response(state)
        
        return {
            'success': True,
            'error': None,
            'sections': sections,
            'task_status': getattr(state, 'task_status', None)
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"{str(e)}\n\n{traceback.format_exc()}",
            'sections': None
        }


def render_response_sections(sections: Dict[str, Any]):
    """Render formatted AI response sections."""
    theme = THEME_CONFIG
    
    # Summary/Main Response
    if sections.get('summary'):
        st.markdown(f"""
        <div style="background: {theme.secondary_bg}; border: 1px solid {theme.glass_border}; 
                    border-radius: {theme.border_radius_lg}; padding: 1.5rem; margin-bottom: 1.5rem;
                    border-left: 4px solid {theme.accent};">
            <div style="color: {theme.accent}; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem; 
                        display: flex; align-items: center; gap: 0.5rem;">
                <span>🤖</span> AI RESPONSE
            </div>
            <div style="color: {theme.text_primary}; line-height: 1.6; white-space: pre-wrap;">
                {sections['summary']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Hypotheses
    if sections.get('hypotheses'):
        st.markdown("#### 🔬 Hypotheses")
        for h in sections['hypotheses']:
            confidence_pct = h['confidence'] * 100
            conf_color = theme.success if confidence_pct >= 70 else theme.warning if confidence_pct >= 50 else theme.danger
            st.markdown(f"""
            <div style="background: {theme.glass_background}; border: 1px solid {theme.glass_border}; 
                        border-radius: 12px; padding: 1rem; margin-bottom: 0.75rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="color: {theme.text_muted}; font-size: 0.75rem;">{h['id']}</span>
                    <span style="background: {conf_color}20; color: {conf_color}; padding: 0.2rem 0.5rem; 
                                 border-radius: 4px; font-size: 0.75rem; font-weight: 600;">{confidence_pct:.0f}% confidence</span>
                </div>
                <div style="color: {theme.text_primary}; font-size: 0.9rem;">{h['description'][:300]}...</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations
    if sections.get('recommendations'):
        st.markdown("#### 📋 Recommendations")
        for r in sections['recommendations']:
            priority_str = r['priority'].value if hasattr(r['priority'], 'value') else str(r['priority']) if r['priority'] else 'Medium'
            priority_color = theme.danger if 'high' in priority_str.lower() or 'critical' in priority_str.lower() else theme.warning
            approval_badge = f"<span style='background: {theme.warning}20; color: {theme.warning}; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem;'>⚠️ Requires Approval</span>" if r.get('requires_approval') else ""
            
            st.markdown(f"""
            <div style="background: {theme.glass_background}; border: 1px solid {theme.glass_border}; 
                        border-radius: 12px; padding: 1rem; margin-bottom: 0.75rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="background: {priority_color}20; color: {priority_color}; padding: 0.2rem 0.5rem; 
                                 border-radius: 4px; font-size: 0.75rem; font-weight: 600;">{priority_str}</span>
                    <span style="color: {theme.text_muted}; font-size: 0.75rem;">~{r['effort_hours']} hrs</span>
                </div>
                <div style="color: {theme.text_primary}; font-weight: 500; margin-bottom: 0.25rem;">{r['action']}</div>
                <div style="color: {theme.text_secondary}; font-size: 0.85rem; margin-bottom: 0.5rem;">{r['impact']}</div>
                <div style="display: flex; gap: 0.5rem; align-items: center;">
                    <span style="color: {theme.text_muted}; font-size: 0.75rem;">👤 {r['responsible_role']}</span>
                    {approval_badge}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Forecasts
    if sections.get('forecasts'):
        st.markdown("#### 📈 Forecasts")
        for f in sections['forecasts']:
            pred_pct = f['prediction'] * 100
            lower_pct = f['lower_bound'] * 100
            upper_pct = f['upper_bound'] * 100
            conf_pct = f['confidence'] * 100
            
            st.markdown(f"""
            <div style="background: {theme.glass_background}; border: 1px solid {theme.glass_border}; 
                        border-radius: 12px; padding: 1rem; margin-bottom: 0.75rem;">
                <div style="color: {theme.accent}; font-weight: 600; margin-bottom: 0.5rem;">
                    {f['metric'].replace('_', ' ').title()}
                </div>
                <div style="display: flex; gap: 2rem; margin-bottom: 0.5rem;">
                    <div>
                        <div style="color: {theme.text_muted}; font-size: 0.7rem;">Prediction</div>
                        <div style="color: {theme.text_primary}; font-size: 1.2rem; font-weight: 700;">{pred_pct:.1f}%</div>
                    </div>
                    <div>
                        <div style="color: {theme.text_muted}; font-size: 0.7rem;">Range</div>
                        <div style="color: {theme.text_secondary}; font-size: 0.9rem;">{lower_pct:.0f}% - {upper_pct:.0f}%</div>
                    </div>
                    <div>
                        <div style="color: {theme.text_muted}; font-size: 0.7rem;">Timeframe</div>
                        <div style="color: {theme.text_secondary}; font-size: 0.9rem;">{f['timeframe']}</div>
                    </div>
                </div>
                <div style="color: {theme.text_muted}; font-size: 0.75rem;">Confidence: {conf_pct:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Communications
    if sections.get('communications'):
        st.markdown("#### 📧 Draft Communications")
        for c in sections['communications']:
            with st.expander(f"📧 {c['subject']}", expanded=False):
                st.markdown(f"**To:** {c['recipient']}")
                st.markdown(f"**Type:** {c['type']}")
                st.markdown("---")
                st.markdown(c['body'])
                if c.get('requires_approval'):
                    st.warning("⚠️ This communication requires human approval before sending.")
                col1, col2 = st.columns(2)
                with col1:
                    st.button("✅ Approve & Send", key=f"approve_{c['id']}", disabled=True)
                with col2:
                    st.button("✏️ Edit", key=f"edit_{c['id']}", disabled=True)
    
    # Errors
    if sections.get('errors'):
        st.markdown("#### ⚠️ Issues Encountered")
        for err in sections['errors']:
            st.warning(err)


def render_page(user: Dict[str, Any]):
    """Render the AI Assistant page with full agent integration."""
    
    theme = THEME_CONFIG
    initialize_chat_state()
    
    # Handle auto-filled example
    default_value = ""
    if st.session_state.selected_example:
        default_value = st.session_state.selected_example
        st.session_state.selected_example = None
    
    # Custom CSS for Chat Interface
    st.markdown(f"""
    <style>
        .chat-container {{
            background: {theme.background};
            border-radius: {theme.border_radius_lg};
            border: 1px solid {theme.glass_border};
            padding: 1.5rem;
            min-height: 200px;
        }}
        .user-bubble {{
            background: {theme.gradient_accent};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 20px 20px 0 20px;
            margin: 1rem 0;
            margin-left: auto;
            max-width: 80%;
            box-shadow: {theme.shadow_md};
        }}
        .ai-loading {{
            background: {theme.secondary_bg};
            border: 1px solid {theme.glass_border};
            padding: 1.5rem;
            border-radius: 16px;
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown(f"""
        <div style="text-align: center; padding: 2rem 0;">
            <div style="display: inline-block; padding: 1.5rem; border-radius: 50%; 
                        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(168, 85, 247, 0.1)); 
                        margin-bottom: 1rem; box-shadow: 0 0 30px rgba(139, 92, 246, 0.3);">
                <div style="font-size: 3.5rem; filter: drop-shadow(0 0 15px {theme.accent});">🤖</div>
            </div>
            <h1 style="color: {theme.text_primary}; margin-bottom: 0.5rem; font-size: 2.2rem;">Nexus AI Assistant</h1>
            <p style="color: {theme.text_secondary}; max-width: 600px; margin: 0 auto; font-size: 1rem;">
                Your intelligent copilot powered by <strong style="color: {theme.accent};">6 AI Agents</strong>. 
                Ask about studies, sites, data quality, forecasts, or request action plans.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Check orchestrator status
    orchestrator = get_orchestrator()
    if orchestrator is None:
        error_msg = st.session_state.get('ai_orchestrator_error', 'Unknown error')
        
        # Provide helpful guidance based on error
        if "groq" in error_msg.lower() or "api" in error_msg.lower():
            st.error(f"""
            ⚠️ **AI System Not Available**
            
            The AI agent needs an LLM provider. Please configure one:
            
            **Option 1: Groq (Cloud - Recommended)**
            1. Get API key from [console.groq.com](https://console.groq.com)
            2. Add to your `.env` file:
            ```
            GROQ_API_KEY=your-key-here
            LLM_PRIMARY_PROVIDER=groq
            ```
            3. Restart the dashboard
            
            **Option 2: Ollama (Local)**
            1. Install Ollama from [ollama.ai](https://ollama.ai)
            2. Run: `ollama serve`
            3. Pull a model: `ollama pull llama3.1:8b`
            
            **Error:** {error_msg}
            """)
        else:
            st.error(f"""
            ⚠️ **AI System Not Available**
            
            The AI agent orchestrator could not be initialized.
            
            **Error:** {error_msg}
            
            Please ensure LLM is configured (Groq or Ollama).
            """)
        
        st.info("You can still browse the dashboard, but AI-powered features are unavailable.")
        return
    else:
        st.success("✅ AI Agent System Online", icon="🤖")
    
    # Main Chat Interface
    st.markdown("### 💬 Ask Nexus")
    
    # Input area
    user_input = st.text_input(
        "Ask a question...",
        value=default_value,
        placeholder="e.g., 'Show me critical sites' or 'Why is Study_21 DQI dropping?'",
        key="ai_chat_input_field",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        send = st.button("📤 SEND", type="primary", use_container_width=True)
    with col2:
        clear = st.button("🗑️ Clear", use_container_width=True)
    
    if clear:
        st.session_state.chat_history = []
        st.rerun()
    
    # Process query
    if send and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Show processing state
        with st.spinner("🤖 AI Agents processing your query... (Supervisor → Diagnostic → Forecaster → Resolver → Synthesizer)"):
            result = process_query(user_input)
        
        if result['success']:
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': result['sections'],
                'timestamp': datetime.now().isoformat()
            })
        else:
            st.session_state.chat_history.append({
                'role': 'error',
                'content': result['error'],
                'timestamp': datetime.now().isoformat()
            })
        
        st.rerun()
    
    # Display chat history
    st.markdown("---")
    
    if st.session_state.chat_history:
        for i, msg in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10 messages
            if msg['role'] == 'user':
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 1rem;">
                    <div style="background: {theme.gradient_accent}; color: white; padding: 1rem 1.5rem; 
                                border-radius: 20px 20px 0 20px; max-width: 80%; box-shadow: {theme.shadow_md};">
                        <div style="font-size: 0.75rem; opacity: 0.8; margin-bottom: 0.25rem;">YOU</div>
                        {msg['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif msg['role'] == 'assistant':
                st.markdown(f"""
                <div style="background: {theme.secondary_bg}; border: 1px solid {theme.glass_border}; 
                            border-radius: 0 20px 20px 20px; padding: 0.5rem; margin-bottom: 1rem;">
                    <div style="color: {theme.accent}; font-size: 0.75rem; font-weight: 600; padding: 0.5rem 1rem;">
                        🤖 AI ASSISTANT RESPONSE
                    </div>
                </div>
                """, unsafe_allow_html=True)
                render_response_sections(msg['content'])
            elif msg['role'] == 'error':
                st.error(f"❌ Error: {msg['content']}")
    else:
        # Empty state
        st.markdown(f"""
        <div style="background: {theme.secondary_bg}; border: 1px dashed {theme.glass_border}; 
                    border-radius: {theme.border_radius_lg}; padding: 3rem; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.3;">💬</div>
            <div style="color: {theme.text_muted}; font-size: 1.1rem;">Start a conversation to unlock AI-powered insights.</div>
            <div style="color: {theme.text_muted}; font-size: 0.85rem; margin-top: 0.5rem;">
                Try one of the quick prompts below, or ask your own question.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Quick Prompts / Examples
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### ⚡ Quick Start Prompts")
    
    examples = [
        ("🚨 Critical Issues", "Show sites with critical DQI scores and recommend actions"),
        ("📈 DB Lock Forecast", "When will we reach database lock readiness? Include timeline and risks"),
        ("🔍 Root Cause", "Investigate why data quality has dropped and form hypotheses"),
        ("📧 Site Communication", "Draft an email to site coordinators about pending queries"),
    ]
    
    cols = st.columns(4)
    for i, (label, prompt) in enumerate(examples):
        with cols[i]:
            if st.button(f"{label}", key=f"ex_{i}", use_container_width=True, help=prompt):
                st.session_state.selected_example = prompt
                st.rerun()

    # Agent Status Section
    st.markdown("---")
    st.markdown("### 🤖 Agent Pipeline")
    
    agent_cols = st.columns(6)
    agents = [
        ("🎯", "Supervisor", "Routes & orchestrates"),
        ("🔬", "Diagnostic", "Investigates issues"),
        ("📊", "Forecaster", "Predicts timelines"),
        ("🔧", "Resolver", "Creates action plans"),
        ("📤", "Executor", "Executes actions"),
        ("📧", "Communicator", "Drafts messages")
    ]
    
    for i, (icon, name, desc) in enumerate(agents):
        with agent_cols[i]:
            st.markdown(f"""
            <div style="background: {theme.glass_background}; border: 1px solid {theme.glass_border}; 
                        border-radius: 12px; padding: 1rem; text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
                <div style="color: {theme.text_primary}; font-weight: 600; font-size: 0.85rem;">{name}</div>
                <div style="color: {theme.text_muted}; font-size: 0.7rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Capabilities Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
        <div style="padding: 1.5rem; background: {theme.glass_background}; border-radius: 12px; 
                    border: 1px solid {theme.glass_border};">
            <h4 style="color: {theme.text_primary}; margin: 0 0 0.5rem 0;">🧠 6-Agent Intelligence</h4>
            <p style="color: {theme.text_muted}; font-size: 0.85rem; margin: 0;">
                Powered by specialized AI agents that collaborate to analyze, diagnose, forecast, and resolve.
            </p>
        </div>
        <div style="padding: 1.5rem; background: {theme.glass_background}; border-radius: 12px; 
                    border: 1px solid {theme.glass_border};">
            <h4 style="color: {theme.text_primary}; margin: 0 0 0.5rem 0;">🛡️ Human-in-the-Loop</h4>
            <p style="color: {theme.text_muted}; font-size: 0.85rem; margin: 0;">
                AI recommends, you decide. All actions require human approval before execution.
            </p>
        </div>
        <div style="padding: 1.5rem; background: {theme.glass_background}; border-radius: 12px; 
                    border: 1px solid {theme.glass_border};">
            <h4 style="color: {theme.text_primary}; margin: 0 0 0.5rem 0;">⚡ Real-time Insights</h4>
            <p style="color: {theme.text_muted}; font-size: 0.85rem; margin: 0;">
                Direct connection to the Nexus data lake for up-to-the-second analysis and recommendations.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)