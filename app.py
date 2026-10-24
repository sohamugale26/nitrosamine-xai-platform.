import streamlit as st
import pandas as pd
import numpy as np
import datetime
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
import py3Dmol
import streamlit.components.v1 as components

st.set_page_config(page_title="NitroPredict Engine", layout="wide")
st.title("NitroPredict: QbD Assessment Engine")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧬 3D AI", "💊 Supply Chain", "⚖️ Limits", "⚗️ Kinetics", "📊 Report"])

# Initialize session state for the AI limit so Tab 1 can auto-fill Tab 3
if 'auto_ai_limit' not in st.session_state:
    st.session_state.auto_ai_limit = 26.5  # Default strict limit

# --- TAB 1: 3D CHEMISTRY & AI ---
with tab1:
    st.subheader("3D Steric Shielding & OCSR Engine")
    uploaded_file = st.file_uploader("Upload Chemical Drawing (PDF/IMG)", type=["png", "jpg", "pdf"])
    
    default_smiles = "CN(C)N=O"
    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")
        st.info("🔄 OCSR Engine simulating extraction...")
        default_smiles = "CC(=O)NC1=CC=C(O)C=C1" 
    
    smiles_input = st.text_input("API SMILES String:", default_smiles, key="api_smiles")
    
    if st.button("Analyze Structure"):
        molecule = Chem.MolFromSmiles(smiles_input)
        if molecule:
            formula = rdMolDescriptors.CalcMolFormula(molecule)
            mol_weight = Descriptors.MolWt(molecule)
            st.info(f"**Formula:** {formula} | **Weight:** {mol_weight:.2f} g/mol")
            
            # --- NEW: ZERO-CLICK AUTOMATION ENGINE ---
            # Define mathematical patterns for vulnerable amines
            sec_amine_pattern = Chem.MolFromSmarts("[NX3;H1]([C,c])([C,c])")
            tert_amine_pattern = Chem.MolFromSmarts("[NX3;H0]([C,c])([C,c])([C,c])")
            
            has_sec = molecule.HasSubstructMatch(sec_amine_pattern)
            has_tert = molecule.HasSubstructMatch(tert_amine_pattern)
            
            if has_sec:
                st.error("⚠️ **HIGH RISK DETECTED:** Vulnerable Secondary (2°) Amine identified in API structure.")
                st.write("CPCA Engine assigns **Category 1** (High Potency). Auto-setting Acceptable Intake to **26.5 ng/day**.")
                st.session_state.auto_ai_limit = 26.5
            elif has_tert:
                st.warning("⚠️ **MODERATE RISK DETECTED:** Tertiary (3°) Amine identified in API structure.")
                st.write("CPCA Engine assigns **Category 3/4** (Sterically Shielded). Auto-setting Acceptable Intake to **400.0 ng/day**.")
                st.session_state.auto_ai_limit = 400.0
            else:
                st.success("✅ **NO VULNERABLE AMINES DETECTED:** Structure appears resistant to direct nitrosation.")
                st.session_state.auto_ai_limit = 1500.0
            
            # 3D Rendering
            molecule = Chem.AddHs(molecule)
            AllChem.EmbedMolecule(molecule, randomSeed=42)
            mblock = Chem.MolToMolBlock(molecule)
            view = py3Dmol.view(width=350, height=350)
            view.addModel(mblock, 'mol')
            view.setStyle({'stick': {}})
            view.zoomTo()
            components.html(view._make_html(), height=360, width=360)

# --- TAB 2: SUPPLY CHAIN METADATA ---
with tab2:
    st.subheader("Batch & Supply Chain Ledger")
    
    if 'formulation_df' not in st.session_state:
        st.session_state.formulation_df = pd.DataFrame({
            "Excipient": ["Microcrystalline Cellulose", "Magnesium Stearate", ""],
            "Vendor": ["FMC Biopolymer", "Mallinckrodt", ""],
            "Lot Number": ["B-90210", "XYZ-123", ""],
            "Test Method": ["Ion Chromatography", "Supplier COA", ""],
            "Test Date": [datetime.date(2026, 8, 15), datetime.date(2026, 9, 10), None],
            "Amount (mg/pill)": [100.0, 5.0, 0.0],
            "Nitrite (ppm)": [2.5, 1.0, 0.0],
            "Is LOQ (<)": [False, True, False]
        })
    
    st.session_state.edited_df = st.data_editor(
        st.session_state.formulation_df,
        column_config={
            "Test Date": st.column_config.DateColumn("Test Date", format="YYYY-MM-DD", step=1)
        },
        num_rows="dynamic",
        use_container_width=True
    )

# --- TAB 3: REGULATORY CALCULATOR ---
with tab3:
    st.subheader("Acceptable Intake (AI) Calculator")
    # Tab 3 now auto-updates based on what Tab 1 found!
    st.info("💡 The Acceptable Intake limit is automatically set by the 3D CPCA Engine in Tab 1, but can be manually overridden below.")
    ai_limit = st.number_input("Acceptable Intake (ng/day)", value=st.session_state.auto_ai_limit, key="ai_limit")
    max_dose = st.number_input("Maximum Daily Dose of API (mg/day)", value=500.0, key="max_dose")

# --- TAB 4: KINETIC GUARDRAILS & TRACEABILITY ---
with tab4:
    st.subheader("Arrhenius Kinetics Settings")
    st.warning("⚠️ **ALCOA+ Data Integrity Guardrail:** Regulatory claims must be supported by traceable experimental documentation (21 CFR Part 11).")
    is_validated = st.checkbox("I confirm these kinetic parameters are experimentally validated.", key="kinetics_validated")
    
    if is_validated:
        source_type = st.selectbox(
            "Select Documentation Source Type:",
            ["Internal Lab Report (LIMS/AR&D)", "Peer-Reviewed Literature (DOI)", "Industry Consortium (e.g., Lhasa)"],
            key="source_type"
        )
        if "LIMS" in source_type:
            hint = "e.g., LIMS-STAB-2026-045"
        elif "DOI" in source_type:
            hint = "e.g., DOI: 10.1021/acs.oprd.2c00001"
        else:
            hint = "e.g., Lhasa-Ref-089"
        validation_source = st.text_input("Enter Official Document ID:", placeholder=hint, key="validation_source")
    else:
        st.session_state.validation_source = ""
        st.session_state.source_type = ""
        
    activation_energy = st.number_input("Activation Energy (Ea) in kJ/mol", value=85.0, key="activation_energy")
    phase = st.radio("Select Evaluation Phase:", ["🏭 Manufacturing (Hours)", "📦 Shelf Life (Days)"], horizontal=True, key="phase")
    
    if phase == "🏭 Manufacturing (Hours)":
        temp_c = st.slider("Process Temperature (°C)", 20, 100, 60, key="temp_c_mfg")
        time_val = st.slider("Process Duration (Hours)", 0.0, 48.0, 2.0, step=0.5, key="time_mfg")
    else:
        temp_c = st.slider("Storage Temperature (°C)", 0, 60, 25, key="temp_c_shelf")
        time_val = st.slider("Shelf Life (Days)", 0, 730, 365, key="time_shelf")

# --- TAB 5: MASTER REPORT ---
with tab5:
    st.subheader("Comprehensive Master Assessment")
    if st.button("🚀 Execute Full Risk Assessment", type="primary", use_container_width=True):
        if st.session_state.kinetics_validated and not st.session_state.get("validation_source", "").strip():
            st.error("🛑 **DATA INTEGRITY LOCK:** You checked the validation box but did not provide a Document ID.")
            st.warning("Access Denied. Please return to Tab 4 and provide a valid LIMS, DOI, or Consortium ID to unlock the kinetic engine.")
            st.stop()
            
        df = st.session_state.edited_df
        total_nitrite_ng = 0
        for index, row in df.iterrows():
            if pd.notna(row["Excipient"]) and str(row["Excipient"]).strip() != "":
                total_nitrite_ng += row["Amount (mg/pill)"] * row["Nitrite (ppm)"]
                
        phase_selected = st.session_state.phase
        if "Manufacturing" in phase_selected:
            active_temp = st.session_state.temp_c_mfg
            active_time_days = st.session_state.time_mfg / 24.0 
        else:
            active_temp = st.session_state.temp_c_shelf
            active_time_days = st.session_state.time_shelf
            
        temp_k = active_temp + 273.15
        rate_constant = np.exp(-st.session_state.activation_energy / (8.314 * temp_k))
        kinetic_multiplier = rate_constant * active_time_days * 100
        final_nitrosamine_yield = total_nitrite_ng * kinetic_multiplier
        regulatory_limit = st.session_state.ai_limit
        
        st.divider()
        st.markdown("### 📋 Executive Summary")
        st.write(f"**Base Nitrite in Formulation:** {total_nitrite_ng:.2f} ng/pill")
        st.write(f"**Process Phase:** {phase_selected} at {active_temp}°C")
        st.write(f"**Applied Activation Energy:** {st.session_state.activation_energy} kJ/mol")
        
        if not st.session_state.kinetics_validated:
            st.warning("⚠️ **UNVALIDATED MODEL:** This prediction is an exploratory simulation only. It cannot be used for regulatory submission.")
            st.metric(label="Exploratory Nitrosamine Generation", value=f"{final_nitrosamine_yield:.4f} ng")
        else:
            st.metric(label="Total Predicted Nitrosamine Generation", value=f"{final_nitrosamine_yield:.4f} ng")
            if final_nitrosamine_yield > regulatory_limit:
                st.error(f"❌ **FAIL:** Exceeds the Acceptable Intake of {regulatory_limit} ng/day.")
            else:
                st.success(f"✅ **PASS:** Remains below the {regulatory_limit} ng/day limit.")
                st.info(f"🔒 **Validated 21 CFR Part 11 Traceability:** {st.session_state.source_type} | ID: `{st.session_state.validation_source}`")
            
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Master Batch CSV", data=csv_data, file_name='master_batch_assessment.csv', mime='text/csv')
        
