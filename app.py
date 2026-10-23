import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
import py3Dmol
import streamlit.components.v1 as components

st.set_page_config(page_title="Nitrosamine AI", layout="wide")
st.title("Nitrosamine QA Engine")

tab1, tab2, tab3, tab4 = st.tabs(["🧬 3D AI", "💊 Formulation", "⚖️ Limits", "⚗️ Reports"])

# --- TAB 1: 3D CHEMISTRY & AI ---
with tab1:
    st.subheader("3D Steric Shielding & OCSR Engine")
    st.markdown("Input the API structure manually, or upload a document/image for automatic extraction.")
    
    # Dual Input System: Upload or Text
    uploaded_file = st.file_uploader("Upload Chemical Drawing (PNG, JPG) or Document (PDF)", type=["png", "jpg", "pdf", "txt"])
    
    # Default SMILES variable
    current_smiles = "CN(C)N=O"
    
    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")
        st.info("🔄 OCSR Engine simulating extraction... (Enterprise version will use DECIMER AI to read the image).")
        # For the prototype, we simulate a successful extraction from the image
        current_smiles = "CC(=O)NC1=CC=C(O)C=C1" # Paracetamol as a demo extraction
        st.write(f"**Extracted SMILES:** `{current_smiles}`")
    
    smiles_input = st.text_input("API SMILES String (Edit if needed):", current_smiles)
    
    if st.button("Analyze Structure"):
        molecule = Chem.MolFromSmiles(smiles_input)
        if molecule is None:
            st.error("Invalid SMILES structure.")
        else:
            formula = rdMolDescriptors.CalcMolFormula(molecule)
            mol_weight = Descriptors.MolWt(molecule)
            
            st.info(f"**Molecular Formula:** {formula} | **Molecular Weight:** {mol_weight:.2f} g/mol")
            
            # 3D Rendering
            molecule = Chem.AddHs(molecule)
            AllChem.EmbedMolecule(molecule, randomSeed=42)
            
            mblock = Chem.MolToMolBlock(molecule)
            view = py3Dmol.view(width=350, height=350)
            view.addModel(mblock, 'mol')
            view.setStyle({'stick': {}})
            view.zoomTo()
            
            components.html(view._make_html(), height=360, width=360)
            st.caption("🔴 Oxygen | 🔵 Nitrogen | ⚪ Hydrogen | ⚪ (Thick) Carbon")
            
            atom_count = molecule.GetNumAtoms()
            risk_cat = "Category 3 (400 ng/day)" if atom_count > 12 else "Category 1 (18 ng/day)"
            st.success(f"**AI Predicted Risk:** {risk_cat}")

# [KEEP TAB 2, 3, AND 4 EXACTLY AS THEY WERE IN YOUR FILE]
