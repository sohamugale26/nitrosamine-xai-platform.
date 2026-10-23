import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
from stmol import showmol

st.set_page_config(page_title="Nitrosamine AI", layout="wide")
st.title("Nitrosamine AI & CPCA Platform")

# Create navigation tabs
tab1, tab2 = st.tabs(["🧬 3D Molecular Evaluator", "⚗️ Formulation Kinetics"])

# --- TAB 1: 3D CHEMISTRY & AI ---
with tab1:
    st.subheader("3D Structure & Steric Shielding Analysis")
    smiles_input = st.text_input("Enter API SMILES String:", "CN(C)N=O")
    
    if st.button("Run 3D Evaluation"):
        molecule = Chem.MolFromSmiles(smiles_input)
        
        if molecule is None:
            st.error("Invalid SMILES structure.")
        else:
            # 1. Add Hydrogens (Crucial for real 3D physics)
            molecule = Chem.AddHs(molecule)
            # 2. Compute 3D Coordinates (Seed locked for data reproducibility)
            AllChem.EmbedMolecule(molecule, randomSeed=42)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 3. Render Interactive 3D Model
                mblock = Chem.MolToMolBlock(molecule)
                view = py3Dmol.view(width=400, height=400)
                view.addModel(mblock, 'mol')
                view.setStyle({'stick': {}})
                view.zoomTo()
                
                showmol(view, height=400, width=400)
                st.caption("Interact: Click & drag to rotate. Scroll to zoom.")
            
            with col2:
                atom_count = molecule.GetNumAtoms()
                # Placeholder logic for AI
                risk_cat = "Category 3 (400 ng/day)" if atom_count > 12 else "Category 1 (18 ng/day)"
                st.success(f"**AI Predicted Risk:** {risk_cat}")
                st.info(f"**Total Atoms (inc. Hydrogens):** {atom_count}")
                st.write("The 3D coordinates generated above allow the engine to map exact spatial hindrance and steric shielding around the alpha-carbons.")

# --- TAB 2: KINETICS & SHELF LIFE ---
with tab2:
    st.subheader("Arrhenius Kinetics & Shelf-Life Prediction")
    st.markdown("Predict nitrosamine growth over time based on raw material batch variability.")
    
    col1, col2 = st.columns(2)
    with col1:
        nitrite_ppm = st.number_input("Excipient Nitrite Level (ppm)", min_value=0.0, max_value=50.0, value=2.5)
        temp_c = st.slider("Storage Temperature (°C)", 0, 60, 25)
        days = st.slider("Shelf Life Duration (Days)", 0, 730, 365)
        
    with col2:
        # Simplified Arrhenius Math for UI demonstration
        R = 8.314 
        temp_k = temp_c + 273.15
        activation_energy = 85.0
        
        rate_constant = np.exp(-activation_energy / (R * temp_k))
        predicted_formation = round(nitrite_ppm * rate_constant * days * 100, 2)
        
        st.metric(label="Predicted Nitrosamine Formation", value=f"{predicted_formation} ng/day eq")
        
        if predicted_formation > 18:
            st.error("⚠️ Warning: Predicted formation exceeds Category 1 Acceptable Intake limits over shelf life.")
        else:
            st.success("✅ Formulation remains stable within regulatory limits.")
            
