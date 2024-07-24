import json
import re
from neo4j import GraphDatabase
from openai import OpenAI
from difflib import SequenceMatcher
import itertools
from tqdm import tqdm

import dotenv
dotenv.load_dotenv()

# Neo4j connection details
URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "ineedastrongerpassword"

client = OpenAI()

def get_nodes_and_neighbors(tx):
    query = """
        MATCH (n)-[r]-(neighbor)
        WHERE n.name IS NOT NULL AND neighbor.name IS NOT NULL
        WITH n, COLLECT(DISTINCT {name: neighbor.name, predicate: type(r)}) AS neighbors
        RETURN n.name AS name, labels(n) as category, neighbors
    """
    result = tx.run(query)
    return {record["name"]: {"name":record["name"], "category": record['category'], "neighbors":record["neighbors"]} for record in result}
    return result

def create_node_description(node_name, neighbors):
    description = f"Node '{node_name}' has the following neighbors:\n"
    for neighbor in neighbors:
        description += f"- {neighbor['name']} (connected by {neighbor['predicate']})\n"
    return description.strip()

def are_names_similar(name1, name2, threshold=0.8):

    category1 = name1['category']
    category2 = name2['category']

    name1 = name1['name']
    name2 = name2['name']

    # print(category1, category2)

    if category1[-1] != category2[-1]:
        return False
    
    if name1 in name2 or name2 in name1:
        return True
    return SequenceMatcher(None, name1, name2).ratio() > threshold

def compare_nodes_with_gpt4(node1_name, node1_neighbors, node2_name, node2_neighbors):
    node1_desc = create_node_description(node1_name, node1_neighbors)
    node2_desc = create_node_description(node2_name, node2_neighbors)
    
    prompt = f"""
    Compare the following two nodes and their neighborhoods:

    Node 1:
    {node1_desc}

    Node 2:
    {node2_desc}

    Are these two nodes similar?

    If you are confident that names are almost the same, then ignore neighbourhood completely. Similarity could be measured by the following:
    - Names are similar in terms of spelling or contain spelling mistakes
    - names that are shortened or abbreviated versions of each other
    - names that contain titles

    But if you're not sure, then consider both the names and the neighborhood structures.

    If yes, briefly explain why they are similar and then give what would be full name of the node, given their names are {node1_name} and {node2_name}? $$<NAME>$$
    If no, briefly explain why they are not similar and then give a final output $$NOT_SIMILAR$$.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "User: " + prompt}
        ],
        max_tokens=512
    )
    
    # import pdb; pdb.set_trace()
    content = response.choices[0].message.content

    print("===================================")
    print("Inptut:", prompt)
    print("++++++++++++++++++++++++++++")
    print("Output:", content)

    if "$$NOT_SIMILAR$$" in content:
        print("MERGING_STATUS_FOR_NODES:", node1_name, " AND ", node2_name, " -> ", "NOT_SIMILAR")
        return None
    else:
        name_match = re.search(r'\$\$(.*?)($|\$\$)', content, re.DOTALL)
        if name_match:
            new_name = name_match.group(1).strip()
            print("MERGING_STATUS_FOR_NODES:", node1_name, " AND ", node2_name, " -> ", new_name)
            return new_name
        else:
            print("MERGING_STATUS_FOR_NODES:", node1_name, " AND ", node2_name, " -> ", "NOT_SIMILAR")
            return None

def merge_nodes(tx, node1_name, node2_name, new_name):
    query = """
    MATCH (n1 {name: $node1_name})
    MATCH (n2 {name: $node2_name})
    WHERE id(n1) < id(n2)
    CALL apoc.merge.node(['Node'], {name: $new_name}, 
                         apoc.map.mergeMaps(n1, n2, {name: $new_name})) YIELD node
    WITH n1, n2, node
    CALL apoc.refactor.mergeNodes([n1, n2, node], {properties: 'combine', mergeRels: true})
    YIELD node as mergedNode
    RETURN mergedNode
    """
    result = tx.run(query, node1_name=node1_name, node2_name=node2_name, new_name=new_name)
    return result.single()['mergedNode']


def main():
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    try:
        with driver.session() as session:
            nodes_and_neighbors = session.read_transaction(get_nodes_and_neighbors)
            print(f"Retrieved {len(nodes_and_neighbors)} nodes.")

            # Create pairs of nodes with similar names
            node_pairs = list(itertools.combinations(nodes_and_neighbors.keys(), 2))
            similar_name_pairs = [(nodes_and_neighbors[n1]['name'], nodes_and_neighbors[n2]['name']) for n1, n2 in node_pairs if are_names_similar(nodes_and_neighbors[n1], nodes_and_neighbors[n2])]

            print(f"Found {len(similar_name_pairs)} pairs of nodes with similar names.")

            for node1_name, node2_name in similar_name_pairs:
                print(f"\nComparing '{node1_name}' and '{node2_name}':")

            new_names = []
            for node1_name, node2_name in tqdm(similar_name_pairs):
                print ( "----------------------------------------------------------------------")
                print(f"\nComparing '{node1_name}' and '{node2_name}':")
                new_name = compare_nodes_with_gpt4(
                    node1_name, nodes_and_neighbors[node1_name]['neighbors'],
                    node2_name, nodes_and_neighbors[node2_name]['neighbors']
                )

                if new_name:
                    print(f"Nodes are similar. Merging with new name: {new_name}")
                    new_names.append({'node1': node1_name, 'node2': node2_name, 'new_name': new_name})
                    # merged_node = session.write_transaction(merge_nodes, node1_name, node2_name, new_name)
                else:
                    print("The nodes are not similar.")

            with open('new_names.txt', 'w') as f:
                json.dump(new_names, f, indent=4)

    finally:
        driver.close()

if __name__ == "__main__":
    main()