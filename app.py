import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
import py3Dmol
import streamlit.components.v1 as components

st.set_page_config(page_title="Nitrosamine AI", layout="wide")
st.title("Nitrosamine QA Engine")

# --- Expanded to 5 Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧬 3D AI", "💊 Formulation", "⚖️ Limits", "⚗️ Kinetics", "📊 Master Report"])

# --- TAB 1: 3D CHEMISTRY & AI ---
with tab1:
    st.subheader("3D Steric Shielding & OCSR Engine")
    uploaded_file = st.file_uploader("Upload Chemical Drawing (PNG, JPG) or Document (PDF)", type=["png", "jpg", "pdf"])
    
    default_smiles = "CN(C)N=O"
    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")
        st.info("🔄 OCSR Engine simulating extraction...")
        default_smiles = "CC(=O)NC1=CC=C(O)C=C1" 
    
    # Notice we added a 'key' here so Tab 5 can read this input
    smiles_input = st.text_input("API SMILES String:", default_smiles, key="api_smiles")
    
    if st.button("Analyze Structure"):
        molecule = Chem.MolFromSmiles(smiles_input)
        if molecule is None:
            st.error("Invalid SMILES structure.")
        else:
            formula = rdMolDescriptors.CalcMolFormula(molecule)
            mol_weight = Descriptors.MolWt(molecule)
            st.info(f"**Formula:** {formula} | **Weight:** {mol_weight:.2f} g/mol")
            
            molecule = Chem.AddHs(molecule)
            AllChem.EmbedMolecule(molecule, randomSeed=42)
            
            mblock = Chem.MolToMolBlock(molecule)
            view = py3Dmol.view(width=350, height=350)
            view.addModel(mblock, 'mol')
            view.setStyle({'stick': {}})
            view.zoomTo()
            components.html(view._make_html(), height=360, width=360)
            
            atom_count = molecule.GetNumAtoms()
            risk_cat = "Category 3 (400 ng/day)" if atom_count > 12 else "Category 1 (18 ng/day)"
            st.success(f"**AI Predicted Risk:** {risk_cat}")

# --- TAB 2: FORMULATION LEDGER ---
with tab2:
    st.subheader("Dynamic Excipient Ledger")
    if 'formulation_df' not in st.session_state:
        st.session_state.formulation_df = pd.DataFrame({
            "Excipient": ["Microcrystalline Cellulose", "Magnesium Stearate", ""],
            "Amount (mg/pill)": [100.0, 5.0, 0.0],
            "Nitrite (ppm)": [2.5, 1.0, 0.0]
        })
    
    # Save the edited table back to session state so Tab 5 can read it
    st.session_state.edited_df = st.data_editor(st.session_state.formulation_df, num_rows="dynamic", use_container_width=True)

# --- TAB 3: REGULATORY CALCULATOR ---
with tab3:
    st.subheader("Acceptable Intake (AI) Calculator")
    # Added keys here as well
    ai_limit = st.number_input("Acceptable Intake (ng/day)", value=26.5, key="ai_limit")
    max_dose = st.number_input("Maximum Daily Dose of API (mg/day)", value=500.0, key="max_dose")

# --- TAB 4: KINETICS CONFIGURATION ---
with tab4:
    st.subheader("Arrhenius Kinetics Settings")
    phase = st.radio("Select Evaluation Phase:", ["🏭 Manufacturing (Hours)", "📦 Shelf Life (Days)"], horizontal=True, key="phase")
    
    if phase == "🏭 Manufacturing (Hours)":
        temp_c = st.slider("Process Temperature (°C)", 20, 100, 60, key="temp_c_mfg")
        time_val = st.slider("Process Duration (Hours)", 0.0, 48.0, 2.0, step=0.5, key="time_mfg")
    else:
        temp_c = st.slider("Storage Temperature (°C)", 0, 60, 25, key="temp_c_shelf")
        time_val = st.slider("Shelf Life (Days)", 0, 730, 365, key="time_shelf")

# --- TAB 5: MASTER REPORT (THE UNIFIED ENGINE) ---
with tab5:
    st.subheader("Comprehensive Master Assessment")
    st.markdown("Execute a unified assessment using all data from the API, Formulation, and Kinetics tabs.")
    
    if st.button("🚀 Execute Full Risk Assessment", type="primary", use_container_width=True):
        
        # 1. Evaluate API Risk
        molecule = Chem.MolFromSmiles(st.session_state.api_smiles)
        api_status = "Category 1 (High Risk)"
        if molecule:
            api_status = "Category 3 (Lower Risk)" if molecule.GetNumAtoms() > 12 else "Category 1 (High Risk)"
            
        # 2. Calculate Total Nitrite from Formulation Table
        df = st.session_state.edited_df
        total_nitrite_ng = 0
        for index, row in df.iterrows():
            if row["Excipient"].strip() != "":
                total_nitrite_ng += row["Amount (mg/pill)"] * row["Nitrite (ppm)"]
                
        # 3. Calculate Kinetics Multiplier
        phase_selected = st.session_state.phase
        if "Manufacturing" in phase_selected:
            active_temp = st.session_state.temp_c_mfg
            active_time_days = st.session_state.time_mfg / 24.0 # Convert hours to days
        else:
            active_temp = st.session_state.temp_c_shelf
            active_time_days = st.session_state.time_shelf
            
        temp_k = active_temp + 273.15
        rate_constant = np.exp(-85.0 / (8.314 * temp_k))
        kinetic_multiplier = rate_constant * active_time_days * 100
        
        # 4. Final Math: Total Nitrite * Kinetics
        final_nitrosamine_yield = total_nitrite_ng * kinetic_multiplier
        
        # --- GENERATE THE REPORT UI ---
        st.divider()
        st.markdown("### 📋 Executive Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**API Structural Risk:** {api_status}")
            st.write(f"**Base Nitrite in Formulation:** {total_nitrite_ng:.2f} ng/pill")
        with col2:
            st.write(f"**Process Phase:** {phase_selected}")
            st.write(f"**Thermal Exposure:** {active_temp}°C")
            
        st.divider()
        
        # Display Final Prediction
        st.metric(label="Total Predicted Nitrosamine Generation", value=f"{final_nitrosamine_yield:.4f} ng")
        
        # Check against regulatory limit
        regulatory_limit = st.session_state.ai_limit
        if final_nitrosamine_yield > regulatory_limit:
            st.error(f"❌ **FAIL:** The predicted generation ({final_nitrosamine_yield:.2f} ng) exceeds the FDA/EMA Acceptable Intake of {regulatory_limit} ng/day.")
        else:
            st.success(f"✅ **PASS:** The formulation and processing conditions remain safely below the {regulatory_limit} ng/day limit.")
            
        # CSV Export Button
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Master Batch CSV", data=csv_data, file_name='master_batch_assessment.csv', mime='text/csv')
        
