from flask import Flask, request, jsonify, render_template
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
import itertools
import joblib
import numpy as np
from rdkit.Chem import AllChem

app = Flask(__name__)

# Google Custom Search API key and engine ID
API_KEY = 'AIzaSyDgPHZfBiFC6pLtToxn2bOx9xNXKWWyFUo'
SEARCH_ENGINE_ID = '42a4007db180d48f3'

# PubChem base URL
PUBCHEM_BASE_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound'

def sanitize_molecule(mol):
    Chem.SanitizeMol(mol)
    return mol

def smile_to_mol(smile_string):
    mol = Chem.MolFromSmiles(smile_string)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    mol = sanitize_molecule(mol)
    return mol

def generate_molecule_variants(smile_string, combinations, selected_carbon_indices):
    variants = []

    # Create the original molecule
    mol = Chem.MolFromSmiles(smile_string)
    mol = sanitize_molecule(mol)

    for combination in combinations:
        # Create a writable copy of the original molecule
        rw_mol = Chem.RWMol(mol)

        # Remove hydrogens attached to the selected carbon atoms
        for idx in selected_carbon_indices:
            rw_mol.GetAtomWithIdx(idx).SetNumExplicitHs(0)

        # Add new atoms and bonds
        for i, element in enumerate(combination):
            atom = Chem.Atom(element)
            atom_idx = rw_mol.AddAtom(atom)
            rw_mol.AddBond(selected_carbon_indices[i], atom_idx, Chem.BondType.SINGLE)

        # Generate the SMILES string for the new molecule
        new_smiles = Chem.MolToSmiles(rw_mol)
        new_smiles = new_smiles.replace('~', '')  # Remove tilde if present

        # Sanitize the new molecule and check its validity
        new_mol = sanitize_molecule(rw_mol.GetMol())
        if is_valid_molecule(new_mol):
            variants.append({
                'smiles': new_smiles,
                'mol': new_mol,
                'combination': combination
            })

    return variants

def generate_variants(smile, selected_carbon_indices, replaceable_elements):
    try:
        mol = smile_to_mol(smile)

        # Ensure we have at least as many carbon indices as elements in the combinations
        if len(selected_carbon_indices) > len(replaceable_elements):
            raise ValueError("The number of replaceable elements must be at least as many as the selected carbon indices.")

        # Generate all possible combinations of replaceable elements for the selected carbon indices
        combinations = list(itertools.product(replaceable_elements, repeat=len(selected_carbon_indices)))
        variants = generate_molecule_variants(smile, combinations, selected_carbon_indices)
        
        # Collect SMILES and corresponding RDKit molecules
        variant_data = []
        for variant in variants:
            new_smiles = variant['smiles']
            mol = variant['mol']
            variant_data.append({
                'smiles': new_smiles,
                'mol': mol
            })

        return variant_data
    
    except Exception as e:
        print(f"Error generating variants: {str(e)}")
        return []

def is_valid_molecule(mol):
    try:
        # Perform sanitization to catch any potential issues
        Chem.SanitizeMol(mol)
        # Optionally, check for reasonable properties
        if mol.GetNumAtoms() > 0 and Descriptors.MolWt(mol) > 0:
            return True
    except:
        return False
    return False

from rdkit.Chem import Draw
from io import BytesIO
import base64

def mol_to_img_base64(mol):
    img = Draw.MolToImage(mol, size=(300, 300))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def filter_valid_smiles(out):
    valid_variants = []
    for variant in out:
        variant = variant.strip()
        mol = Chem.MolFromSmiles(variant)
        if mol and is_valid_molecule(mol):
            valid_variants.append(variant)

    return valid_variants


from rdkit import Chem
from rdkit.Chem import Descriptors
import joblib
import numpy as np

# Load the RF model
rf_model = joblib.load("rf_model2.pkl")

# Define a function to predict pIC50 for a given SMILES using the RF model
def featurize_smiles(smiles, n_features=2057):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_features)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_features)
    return np.array(fp)

# Define a function to predict pIC50 for a given SMILES using the RF model
def predict_pic50(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    
    # Convert SMILES to a feature vector of 2057 bits
    input_data = featurize_smiles(smiles).reshape(1, -1)
    
    try:
        # Predict pIC50 using the RF model
        pIC50 = rf_model.predict(input_data)[0]  
    except Exception as e:
        # Handle any exceptions gracefully
        print(f"Prediction error for SMILES '{smiles}': {str(e)}")
        pIC50 = None  # or any default value or error handling as needed
    
    return pIC50

# Define a function to calculate descriptors including ADMET and drug-likeness descriptors
def calculate_descriptors_for_variants(variants_data):
    descriptors_data = []
    
    for idx, variant_data in enumerate(variants_data, start=1):
        smile = variant_data['smiles']
        mol = Chem.MolFromSmiles(smile)

        if mol:
            # Generate the image
            img = Draw.MolToImage(mol)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        if mol is None:
            # Handle invalid SMILES gracefully
            descriptors = {
                'smiles': smile,
                'molecular_weight': None,
                'logP': None,
                'num_atoms': None,
                'num_bonds': None,
                'num_rotatable_bonds': None,
                'num_h_donors': None,
                'num_h_acceptors': None,
                'tpsa': None,
                'aromatic_rings_count': None,
                'pIC50': None,
                'drug_likeness': 'Invalid SMILES'
            }
            descriptors_data.append(descriptors)
            continue
        
        # Calculate RDKit descriptors
        molecular_weight = Descriptors.MolWt(mol)
        logP = Descriptors.MolLogP(mol)
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        num_h_donors = Descriptors.NumHDonors(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        aromatic_rings_count = Descriptors.NumAromaticRings(mol)
        pIC50 = predict_pic50(smile)
        


        # Calculate Lipinski's Rule of Five descriptors for drug-likeness
        violations = 0
        if molecular_weight > 500:
            violations += 1
        if logP > 5:
            violations += 1
        if num_h_donors > 5:
            violations += 1
        if num_h_acceptors > 10:
            violations += 1
        
        # Drug-likeness prediction based on Lipinski's Rule of Five
        if violations <= 1:
            drug_likeness = 'Likely'
        else:
            drug_likeness = 'Not likely'
        

        
        # Combine all descriptors into a dictionary
        descriptors = {
            'smiles': smile,
            'molecular_weight': molecular_weight,
            'logP': logP,
            'num_atoms': num_atoms,
            'num_bonds': num_bonds,
            'num_rotatable_bonds': num_rotatable_bonds,
            'num_h_donors': num_h_donors,
            'num_h_acceptors': num_h_acceptors,
            'tpsa': tpsa,
            'aromatic_rings_count': aromatic_rings_count,
            'pIC50': pIC50,
            'drug_likeness': drug_likeness,
            'image': img_str
        }
        
        descriptors_data.append(descriptors)
        
    
    return descriptors_data
    
import pandas as pd

def save_variants_to_csv(variants_data, file_path):
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(variants_data)

    # Save DataFrame to CSV
    df.to_csv(file_path, index=False)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_drugs', methods=['POST'])
def search_drugs():
    data = request.json
    disease_name = data['disease_name']

    service_url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'key': API_KEY,
        'cx': SEARCH_ENGINE_ID,
        'q': f'{disease_name} drugs',
        'start': 1  # Starting point for pagination
    }

    all_drugs = set()  # Set to collect unique drug names
    max_results = 100  # Max results per page
    results_fetched = 0  # Number of results fetched

    while results_fetched < max_results:
        response = requests.get(service_url, params=params)

        if response.status_code == 200:
            result = response.json()
            if 'items' in result:
                for item in result['items']:
                    title = item.get('title', '').split()[0].strip()  # Extracting only the first word and stripping whitespace
                    if title:
                        # Remove any colon that might still be present after splitting
                        title = title.split(':')[0].strip()
                        all_drugs.add(title)

                results_fetched += len(result['items'])
                params['start'] = results_fetched + 1  # Update start for the next page
            else:
                break  # No more results to fetch
        else:
            return jsonify({'drugs': [], 'message': f'Error in request: {response.status_code} - {response.text}'})

    # Convert set to list for JSON serialization and maintain the order
    drugs = list(all_drugs)
    return jsonify({'drugs': drugs})

@app.route('/get_smile', methods=['POST'])
def get_smile():
    try:
        data = request.json
        selected_drug = data.get('selected_drug')

        if not selected_drug:
            return jsonify({'error': "'selected_drug' key is missing in request data."}), 400

        search_url = f'{PUBCHEM_BASE_URL}/name/{selected_drug}/property/CanonicalSMILES/JSON'
        response = requests.get(search_url, timeout=10)

        if response.status_code == 200:
            result = response.json()
            properties = result.get('PropertyTable', {}).get('Properties', [])
            if properties:
                smiles = properties[0].get('CanonicalSMILES')
                if smiles:
                    # Generate molecule image with numbered carbon atoms
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        for atom in mol.GetAtoms():
                            if atom.GetAtomicNum() == 6:  # Carbon atom
                                atom.SetProp('atomNote', str(atom.GetIdx()))
                        img = Draw.MolToImage(mol, size=(300, 300))

                        # Save image to a file
                        img_path = f'static/{selected_drug}_mol.png'
                        img.save(img_path)

                        return jsonify({'smile': smiles, 'image_url': img_path})
                    else:
                        return jsonify({'error': 'Invalid SMILES string'}), 400
                else:
                    return jsonify({'error': 'SMILES not found in the response'}), 404
            else:
                return jsonify({'error': 'Properties not found in the response'}), 404
        else:
            return jsonify({'error': f'Error fetching SMILES: {response.status_code}'}), response.status_code

    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timed out'}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({'error': 'Error fetching SMILES: ' + str(e)}), 500
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred: ' + str(e)}), 500

    
 # Route to filter variants
@app.route('/filter_variants', methods=['POST'])
def filter_variants_route():
    try:
        data = request.json
        variants = data.get('variants', [])

        if not variants:
            return jsonify({'error': "'variants' key is missing or empty in request data."}), 400

        # Filter out non-SMILES entries from variants
        valid_variants = filter_valid_smiles(variants)
        num_valid_variants = len(valid_variants)

        response_data = {
            'num_valid_variants': num_valid_variants,
            'valid_variants': valid_variants
        }

        # Print valid SMILES for debugging or logging purposes
        print("Valid SMILES:")
        for variant in valid_variants:
            print(variant)

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    # Function to generate descriptors
import os

@app.route('/generate_variants', methods=['POST'])
def generate_variants_route():
    try:
        data = request.json
        smile = data.get('smile')
        carbon_indices = data.get('carbon_indices', [])  # Get user-specified carbon indices
        elements = data.get('elements', [])  # Get user-specified elements

        if not smile or not carbon_indices or not elements:
            return jsonify({'error': "'smile', 'carbon_indices', and 'elements' keys are required in request data."}), 400

        combinations = list(itertools.product(elements, repeat=len(carbon_indices)))
        variants_data = generate_molecule_variants(smile, combinations, carbon_indices)
        descriptors_data = calculate_descriptors_for_variants(variants_data)

        # Save the variants and descriptors to a CSV file
        file_path = 'static/variants_descriptors.csv'
        save_variants_to_csv(descriptors_data, file_path)

        response_data = {
            'num_variants': len(variants_data),
            'variants': descriptors_data,
            'download_link': file_path  # Provide a link to download the CSV file
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
from flask import send_from_directory

@app.route('/download_variants', methods=['GET'])
def download_variants():
    file_path = 'static/variants_descriptors.csv'
    if os.path.exists(file_path):
        return send_from_directory(directory='static', filename='variants_descriptors.csv', as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



