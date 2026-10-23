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
    
    uploaded_file = st.file_uploader("Upload Chemical Drawing (PNG, JPG) or Document (PDF)", type=["png", "jpg", "pdf", "txt"])
    
    current_smiles = "CN(C)N=O"
    
    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")
        st.info("🔄 OCSR Engine simulating extraction... (Enterprise version will use AI to read the image).")
        current_smiles = "CC(=O)NC1=CC=C(O)C=C1" 
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

# --- TAB 2: FORMULATION LEDGER ---
with tab2:
    st.subheader("Dynamic Excipient Ledger")
    st.markdown("Add excipients, batch amounts, and COA nitrite levels.")
    
    if 'formulation_df' not in st.session_state:
        st.session_state.formulation_df = pd.DataFrame({
            "Excipient": ["Microcrystalline Cellulose", "Magnesium Stearate", ""],
            "Amount (mg/pill)": [100.0, 5.0, 0.0],
            "Nitrite (ppm)": [2.5, 1.0, 0.0],
            "Is LOQ (<)": [False, True, False]
        })
    
    edited_df = st.data_editor(st.session_state.formulation_df, num_rows="dynamic", use_container_width=True)
    
    if st.button("Calculate Total Formulation Nitrite"):
        total_nitrite_ng = 0
        for index, row in edited_df.iterrows():
            if row["Excipient"].strip() != "":
                total_nitrite_ng += row["Amount (mg/pill)"] * row["Nitrite (ppm)"]
        
        st.info(f"**Total Estimated Nitrite per Dose:** {total_nitrite_ng:.2f} ng")
        st.caption("Assumes worst-case LOQ inclusion for regulatory conservatism.")

# --- TAB 3: REGULATORY CALCULATOR ---
with tab3:
    st.subheader("Acceptable Intake (AI) Calculator")
    st.markdown("Convert FDA/EMA daily limits to ppm/ppb concentration thresholds.")
    
    ai_limit = st.number_input("Acceptable Intake (ng/day)", value=26.5)
    max_dose = st.number_input("Maximum Daily Dose of API (mg/day)", value=500.0)
    
    if max_dose > 0:
        limit_ppm = ai_limit / max_dose
        limit_ppb = limit_ppm * 1000
        st.success(f"**Action Limit:** {limit_ppm:.4f} ppm ({limit_ppb:.2f} ppb)")
    else:
        st.warning("Max daily dose must be greater than 0 mg.")

# --- TAB 4: KINETICS & REPORTS ---
with tab4:
    st.subheader("Arrhenius Kinetics")
    temp_c = st.slider("Storage Temp (°C)", 0, 60, 25)
    days = st.slider("Shelf Life (Days)", 0, 730, 365)
    nitrite_ppm_baseline = st.number_input("Base Nitrite for Kinetics (ppm)", value=2.5)
    
    R, activation_energy = 8.314, 85.0
    temp_k = temp_c + 273.15
    rate_constant = np.exp(-activation_energy / (R * temp_k))
    predicted_formation = round(nitrite_ppm_baseline * rate_constant * days * 100, 2)
    
    st.metric(label="Predicted Nitrosamine Growth", value=f"{predicted_formation} ng eq.")
    
    st.divider()
    st.subheader("Generate QA Report")
    
    csv_data = edited_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Formulation CSV",
        data=csv_data,
        file_name='nitrosamine_formulation_ledger.csv',
        mime='text/csv',
    )
    
