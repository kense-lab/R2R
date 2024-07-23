import psycopg2
import os
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
        WITH n, COLLECT(DISTINCT {name: neighbor.name, fragment_id: properties(r).fragment_id, predicate: type(r)}) AS neighbors
        RETURN n.name AS name, labels(n) as category, neighbors
    """
    result = tx.run(query)
    return {record["name"]: {"name":record["name"], "category": record['category'], "neighbors":record["neighbors"]} for record in result}
    return result

def create_node_description(node_name, neighbors, chunks=None, max_neighbors=0):
    
    
    if max_neighbors > 0:
        description = f"Node '{node_name}' has the following neighbors:\n"
        for neighbor in neighbors:
            description += f"- {neighbor['name']} (connected by {neighbor['predicate']})\n" 
    else:
        description = f"Node '{node_name}'"

    if chunks:
        description += "\n\n Following are the chunks for the given node :\n"
        for chunk_text in chunks:
            description += f"- {chunk_text}\n"

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

    raise
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

def describe_nodes_with_gpt4(node1_name, node1_neighbors, node1_chunks):
    node1_desc = create_node_description(node1_name, node1_neighbors, node1_chunks)
    
    prompt = f"""
    Describe the following node based on the following chunks. Only consider information that is present in chunks.

    Node:
    {node1_desc}

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

    return content


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

def flatten(lst):
    """Flattens a list of lists into a single list."""
    return [item for sublist in lst for item in sublist]


def get_psql_cur(): 
    
    conn = psycopg2.connect(
        dbname=os.getenv('POSTGRES_DBNAME'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        host=os.getenv('POSTGRES_HOST'),
        port=os.getenv('POSTGRES_PORT')
    )

    cur = conn.cursor()
    return cur


def run_psql_query(cur, query, params):
    cur.execute(query, params)
    return cur.fetchall()

def close_psql(cur):
    cur.close()

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("--pairs", action="store_true", help="Compare pairs of nodes with similar names.")
args = argparser.parse_args()

def main():
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    try:
        with driver.session() as session:
            nodes_and_neighbors = session.read_transaction(get_nodes_and_neighbors)
            print(f"Retrieved {len(nodes_and_neighbors)} nodes.")

            # Create pairs of nodes with similar names
            if args.pairs:
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
            else:

                node_dict = {}
                for node_name, node_data in nodes_and_neighbors.items():

                    if 'Robert' in node_name:

                        print(f"\nDescribing '{node_name}':")
                        fragments = [n['fragment_id'] for n in node_data['neighbors']]
                        fragments = list(set(flatten(fragments)))
                        # source from supabase

                        fragments = [str(f) for f in fragments]

                        query = "SELECT id, metadata FROM vecs.demo_vecs_v143 WHERE id = ANY(%s)"
                        
                        with get_psql_cur() as cur:
                            cur = get_psql_cur()
                            fragments = run_psql_query(cur, query, (fragments,))
                            fragments = [f[1]['text'] for f in fragments]

                        description = describe_nodes_with_gpt4(node_name, node_data['neighbors'], fragments)
                        node_dict[node_name] = {"neighbors":node_data['neighbors'], "fragments":fragments, "description":description}

                        # embedding 
                        with open('node_descriptions_wo_nbd.txt', 'w') as f:
                            json.dump(node_dict, f, indent=4)

                        
                        print(f"Description: {description}")

            with open('new_names.txt', 'w') as f:
                json.dump(new_names, f, indent=4)

    finally:
        driver.close()

if __name__ == "__main__":
    main()