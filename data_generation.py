import networkx as nx
import pandas as pd
import numpy as np

def create_healthcare_knowledge_graph(disease_select, genetic_variants):
    G = nx.DiGraph()

    # Add diseases
    for disease in disease_select:
        G.add_node(disease, type='disease')

    # Add genetic variants
    for variant in genetic_variants:
        G.add_node(variant, type='genetic_variant')

    # Example connections
    for disease in disease_select:
        for variant in genetic_variants:
            G.add_edge(variant, disease, weight=np.random.uniform(0.5, 1.0))

    return G

def apply_semantic_rules(patient_data):
    facts = {}
    genetic_variant = patient_data.get('Genetic_Variant')

    if genetic_variant == 'Mutation X':
        facts['Disease_Risk'] = 'High Risk'
    elif genetic_variant == 'Mutation Y':
        facts['Disease_Risk'] = 'Medium Risk'
    else:
        facts['Disease_Risk'] = 'Low Risk'

    return facts

def generate_synthetic_data(num_samples, G):
    data = []
    diseases = [n for n, attr in G.nodes(data=True) if attr['type'] == 'disease']
    genetic_variants = [n for n, attr in G.nodes(data=True) if attr['type'] == 'genetic_variant']

    for i in range(num_samples):
        patient = {}
        patient['Patient_ID'] = i + 1
        patient['Age'] = np.random.randint(0, 100)
        patient['Gender'] = np.random.choice(['Male', 'Female'])
        patient['Genetic_Variant'] = np.random.choice(genetic_variants)

        # Apply semantic rules
        patient_facts = apply_semantic_rules({'Genetic_Variant': patient['Genetic_Variant']})
        patient.update(patient_facts)

        # Assign disease based on knowledge graph probabilities
        disease_probs = []
        for disease in diseases:
            edge_data = G.get_edge_data(patient['Genetic_Variant'], disease, default=None)
            if edge_data:
                disease_probs.append(edge_data['weight'])
            else:
                disease_probs.append(0.0)

        # Normalize probabilities if sum is not zero
        total_weight = sum(disease_probs)
        if total_weight > 0:
            disease_probs = [prob / total_weight for prob in disease_probs]
        else:
            disease_probs = [1 / len(diseases)] * len(diseases)

        patient['Disease'] = np.random.choice(diseases, p=disease_probs)

        # Simulate other clinical data
        patient['Risk_Score'] = np.random.uniform(0, 1)
        patient['Lab_Result_1'] = np.random.normal(100, 15)
        patient['Lab_Result_2'] = np.random.normal(50, 10)

        data.append(patient)

    synthetic_data = pd.DataFrame(data)
    return synthetic_data
