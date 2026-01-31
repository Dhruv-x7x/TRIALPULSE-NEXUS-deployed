
import logging
from typing import List, Dict, Any, Optional
from src.knowledge.neo4j_graph import get_graph_service

logger = logging.getLogger(__name__)

class NexusKnowledgeGraph:
    """
    Knowledge graph service linking Studies, Sites, Patients, and Issues using Neo4j.
    Provides graph-based traversal for cascade analysis.
    """
    
    def __init__(self, db_manager=None):
        self.graph = get_graph_service()

    def get_entity_connections(self, entity_id: str, entity_type: str) -> List[Dict[str, Any]]:
        """Get connected entities (edges) in the trial graph using Neo4j."""
        self.graph._ensure_connected()
        connections = []
        
        if self.graph.uses_mock:
            # Fallback to mock behavior if Neo4j is unavailable
            return [{"target": "mock-target", "type": "MOCK_REL", "direction": "out"}]

        try:
            with self.graph._driver.session() as session:
                # Dynamic Cypher query based on entity type
                query = f"""
                MATCH (n:{entity_type.capitalize()} {{id: $id}})-[r]-(m)
                RETURN m.id as target, type(r) as type, 
                       CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END as direction,
                       labels(m)[0] as target_type
                LIMIT 50
                """
                result = session.run(query, id=entity_id)
                for record in result:
                    connections.append({
                        "target": record["target"],
                        "type": record["type"],
                        "direction": record["direction"],
                        "target_type": record["target_type"]
                    })
            
            return connections
        except Exception as e:
            logger.error(f"Graph traversal failed for {entity_type}:{entity_id}: {e}")
            return []

    def find_cascade_path(self, start_issue_id: str) -> Dict[str, Any]:
        """Find how an issue cascades through the graph using Neo4j pathfinding."""
        try:
            path_info = self.graph.get_cascade_path(start_issue_id)
            return {
                "path": [node["id"] for node in path_info.path_nodes],
                "impact_radius": path_info.total_impact,
                "blocked_entities": path_info.affected_patients,
                "unlocked_actions": path_info.unlocked_actions
            }
        except Exception as e:
            logger.error(f"Cascade analysis failed for issue {start_issue_id}: {e}")
            return {
                "path": [start_issue_id],
                "impact_radius": 0,
                "blocked_entities": 0,
                "unlocked_actions": []
            }
