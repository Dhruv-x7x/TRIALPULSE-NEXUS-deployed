# File: src/run_rag_knowledge_base.py
"""
Runner script for Phase 4.3: RAG Knowledge Base v1.1
Builds and tests the complete RAG system with improved chunking.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge.rag_knowledge_base import (
    RAGKnowledgeBase,
    RetrievalChain,
    TopicSearch,
    build_rag_knowledge_base
)


def test_search(kb: RAGKnowledgeBase) -> None:
    """Test basic search functionality."""
    print("\n" + "=" * 60)
    print("TESTING SEARCH FUNCTIONALITY")
    print("=" * 60)
    
    test_queries = [
        "What are the requirements for informed consent?",
        "How should serious adverse events be reported?",
        "What is the DQI and how is it calculated?",
        "What are the responsibilities of a CRA during monitoring?",
        "What documents are required for database lock?",
        "How should protocol deviations be handled?",
        "What are the requirements for audit trails?",
        "What is source data verification?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        results = kb.search(query, top_k=3, min_score=0.3)
        
        if results:
            print(f"   Found {len(results)} relevant results:")
            for i, result in enumerate(results, 1):
                print(f"   {i}. [{result['source']}] {result.get('title', 'N/A')[:50]}...")
                print(f"      Score: {result['score']:.3f} | Category: {result.get('category', 'N/A')}")
                print(f"      Preview: {result['text'][:80]}...")
        else:
            print("   No results found with score >= 0.3")


def test_source_filtering(kb: RAGKnowledgeBase) -> None:
    """Test source-based filtering."""
    print("\n" + "=" * 60)
    print("TESTING SOURCE FILTERING")
    print("=" * 60)
    
    query = "adverse event reporting"
    
    for source in ['ich_gcp', 'protocol', 'sop']:
        print(f"\n🔍 Source: {source}")
        results = kb.search(query, top_k=2, source_filter=source)
        
        for result in results:
            print(f"   - {result.get('title', 'N/A')[:50]}... (Score: {result['score']:.3f})")


def test_category_filtering(kb: RAGKnowledgeBase) -> None:
    """Test category-based filtering."""
    print("\n" + "=" * 60)
    print("TESTING CATEGORY FILTERING")
    print("=" * 60)
    
    categories = ['data_quality', 'safety', 'monitoring', 'database_lock']
    query = "procedures and requirements"
    
    for category in categories:
        print(f"\n🔍 Category: {category}")
        results = kb.search(query, top_k=2, category_filter=category)
        
        if results:
            for result in results:
                print(f"   - [{result['source']}] {result.get('title', 'N/A')[:40]}... (Score: {result['score']:.3f})")
        else:
            print("   No results found")


def test_retrieval_chain(kb: RAGKnowledgeBase) -> None:
    """Test the retrieval chain for RAG."""
    print("\n" + "=" * 60)
    print("TESTING RETRIEVAL CHAIN")
    print("=" * 60)
    
    chain = RetrievalChain(kb)
    
    question = "What are the key steps for database lock preparation?"
    
    print(f"\n📝 Question: {question}")
    
    result = chain.query(question, top_k=3)
    
    print(f"\n📚 Retrieved {result['retrieval_count']} sources:")
    for source in result.get('sources', []):
        print(f"   - {source['source']}: {source['title'][:40]}... (Score: {source['score']:.3f})")
        if source.get('category'):
            print(f"     Category: {source['category']}")
    
    print(f"\n📄 Generated Prompt (User) - First 500 chars:")
    print("-" * 40)
    user_prompt = result['prompt']['user']
    print(user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt)


def test_topic_search(kb: RAGKnowledgeBase) -> None:
    """Test topic-based search."""
    print("\n" + "=" * 60)
    print("TESTING TOPIC SEARCH")
    print("=" * 60)
    
    topic_search = TopicSearch(kb)
    
    # List available topics
    print("\n📋 Available Topics:")
    for topic in topic_search.get_available_topics():
        print(f"   - {topic['key']}: {topic['name']}")
    
    # Test a few topics
    test_topics = ['safety_reporting', 'data_quality', 'database_lock']
    
    for topic_key in test_topics:
        print(f"\n🔍 Topic: {topic_key}")
        result = topic_search.search_topic(topic_key, top_k=3)
        
        print(f"   Found {result['total_found']} documents for '{result['topic_name']}'")
        for r in result['results'][:3]:
            print(f"   - [{r['source']}] {r.get('title', 'N/A')[:40]}... (Score: {r['score']:.3f})")
            if r.get('category'):
                print(f"     Category: {r['category']}")


def test_context_retrieval(kb: RAGKnowledgeBase) -> None:
    """Test context retrieval for LLM input."""
    print("\n" + "=" * 60)
    print("TESTING CONTEXT RETRIEVAL")
    print("=" * 60)
    
    query = "How do I resolve open queries before database lock?"
    
    print(f"\n📝 Query: {query}")
    
    context = kb.get_relevant_context(query, max_tokens=1000)
    
    print(f"\n📄 Retrieved Context ({len(context)} characters):")
    print("-" * 40)
    print(context[:800] + "..." if len(context) > 800 else context)


def save_test_results(kb: RAGKnowledgeBase, output_dir: Path) -> None:
    """Save test results and examples."""
    print("\n" + "=" * 60)
    print("SAVING TEST RESULTS")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Example queries with results
    example_queries = [
        "What are the requirements for informed consent?",
        "How should SAEs be reported?",
        "What is the DQI?",
        "What are CRA responsibilities?",
        "How to prepare for database lock?"
    ]
    
    examples = []
    for query in example_queries:
        results = kb.search(query, top_k=3)
        examples.append({
            'query': query,
            'results': [
                {
                    'source': r['source'],
                    'title': r.get('title', ''),
                    'category': r.get('category', ''),
                    'score': r['score'],
                    'text_preview': r['text'][:200]
                }
                for r in results
            ]
        })
    
    # Save examples
    examples_path = output_dir / "rag_search_examples.json"
    with open(examples_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Examples saved to {examples_path}")
    
    # Save statistics
    stats = kb.get_statistics()
    stats['timestamp'] = datetime.now().isoformat()
    stats['version'] = '1.1'
    
    stats_path = output_dir / "rag_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Statistics saved to {stats_path}")


def validate_chunking(kb: RAGKnowledgeBase) -> None:
    """Validate that chunking is balanced."""
    print("\n" + "=" * 60)
    print("VALIDATING CHUNKING BALANCE")
    print("=" * 60)
    
    stats = kb.get_statistics()
    
    # Check source balance
    sources = stats['source_distribution']
    total = sum(sources.values())
    
    print("\n📊 Source Balance:")
    for source, count in sources.items():
        pct = count / total * 100
        status = "✅" if count >= 50 else "⚠️" if count >= 20 else "❌"
        print(f"   {status} {source}: {count} chunks ({pct:.1f}%)")
    
    # Check chunk size
    avg_size = stats['avg_chunk_size']
    min_size = stats['min_chunk_size']
    max_size = stats['max_chunk_size']
    
    print(f"\n📏 Chunk Sizes:")
    status = "✅" if 250 <= avg_size <= 600 else "⚠️"
    print(f"   {status} Average: {avg_size} chars (target: 300-500)")
    print(f"   Min: {min_size} chars")
    print(f"   Max: {max_size} chars")
    
    # Check category coverage
    categories = stats['category_distribution']
    print(f"\n📂 Category Coverage ({len(categories)} categories):")
    
    important_categories = ['data_quality', 'safety', 'monitoring', 'data_management', 'database_lock']
    for cat in important_categories:
        count = categories.get(cat, 0)
        status = "✅" if count >= 5 else "⚠️" if count >= 2 else "❌"
        print(f"   {status} {cat}: {count} chunks")


def main():
    """Main execution function."""
    print("=" * 70)
    print("TRIALPULSE NEXUS 10X - PHASE 4.3: RAG KNOWLEDGE BASE v1.1")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Output directory
    output_dir = Path("data/processed/knowledge/rag")
    
    # Force rebuild to apply fixes
    force_rebuild = True
    
    try:
        kb = RAGKnowledgeBase(knowledge_dir=str(output_dir))
        
        if not force_rebuild and kb.load_index():
            print("\n✅ Loaded existing index")
        else:
            print("\n🔄 Building new index with improved chunking...")
            kb = build_rag_knowledge_base(output_dir=str(output_dir), save_index=True)
        
        # Validate chunking
        validate_chunking(kb)
        
        # Run tests
        test_search(kb)
        test_source_filtering(kb)
        test_category_filtering(kb)
        test_retrieval_chain(kb)
        test_topic_search(kb)
        test_context_retrieval(kb)
        
        # Save test results
        save_test_results(kb, output_dir)
        
        # Print summary
        print("\n" + "=" * 70)
        print("PHASE 4.3 COMPLETE - RAG KNOWLEDGE BASE v1.1")
        print("=" * 70)
        
        stats = kb.get_statistics()
        print(f"""
✅ RAG Knowledge Base Successfully Built!

SUMMARY:
├── Total Documents: {stats['total_documents']}
├── Total Words: {stats['total_words']:,}
├── Chunk Sizes: min={stats['min_chunk_size']}, avg={stats['avg_chunk_size']}, max={stats['max_chunk_size']}
├── Embedding Dimension: {stats['embedding_dimension']}
├── Index Status: {'Ready' if stats['index_built'] else 'Not Built'}
│
├── Sources:
│   ├── ICH-GCP: {stats['source_distribution'].get('ich_gcp', 0)} chunks
│   ├── Protocol KB: {stats['source_distribution'].get('protocol', 0)} chunks
│   └── SOPs: {stats['source_distribution'].get('sop', 0)} chunks
│
├── Top Categories:""")
        
        # Show top categories
        sorted_cats = sorted(stats['category_distribution'].items(), key=lambda x: -x[1])[:5]
        for cat, count in sorted_cats:
            print(f"│   ├── {cat}: {count} chunks")
        
        print(f"""│
└── Output Files:
    ├── {output_dir}/rag_embeddings.npy
    ├── {output_dir}/rag_documents.json
    ├── {output_dir}/rag_metadata.json
    ├── {output_dir}/rag_search_examples.json
    └── {output_dir}/rag_statistics.json

CAPABILITIES:
✓ Semantic search across all knowledge sources
✓ Source-based filtering (ich_gcp, protocol, sop)
✓ Category-based filtering
✓ Topic-based search (10 predefined topics)
✓ Context retrieval for LLM input
✓ Retrieval chain for RAG applications
✓ Persistent index (save/load)
""")
        
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())