from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict

# Neo4j connection details
URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "ineedastrongerpassword"

def get_entity_names(tx):
    query = "MATCH (n) WHERE n.name IS NOT NULL RETURN n.name AS name"
    result = tx.run(query)
    return [record["name"] for record in result]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_cluster_names(names, threshold=0.8):
    # Load pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings
    embeddings = model.encode(names)
    
    # Initialize clusters
    clusters = defaultdict(list)
    cluster_id = 0
    
    for i, name in enumerate(names):
        if all(cosine_similarity(embeddings[i], embeddings[j]) < threshold 
               for j, cluster in clusters.items()):
            # Create new cluster
            clusters[cluster_id] = [i]
            cluster_id += 1
        else:
            # Add to existing cluster
            max_similarity = -1
            best_cluster = -1
            for j, cluster in clusters.items():
                similarity = max(cosine_similarity(embeddings[i], embeddings[k]) 
                                 for k in cluster)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_cluster = j
            clusters[best_cluster].append(i)
    
    # Convert indices back to names
    named_clusters = {k: [names[i] for i in v] for k, v in clusters.items()}
    return named_clusters

def main():
    # Connect to Neo4j
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    try:
        with driver.session() as session:
            # Get entity names
            names = session.read_transaction(get_entity_names)
            print(f"Retrieved {len(names)} entity names.")

            # Perform clustering
            clusters = get_cluster_names(names)

            # Print results
            for i, cluster_names in clusters.items():
                print(f"\nCluster {i}:")
                for name in cluster_names[:10]:  # Print first 10 names in each cluster
                    print(f"  - {name}")
                if len(cluster_names) > 10:
                    print(f"  ... and {len(cluster_names) - 10} more")

            print(f"\nTotal clusters found: {len(clusters)}")

    finally:
        driver.close()

if __name__ == "__main__":
    main()