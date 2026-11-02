import streamlit as st
import pandas as pd
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
import py3Dmol
import streamlit.components.v1 as components

st.set_page_config(page_title="NitroPredict: In-Silico Risk Prioritization", layout="wide")
st.title("NitroPredict: In-Silico Forced Degradation Engine")

# The Recommended Thesis Positioning Statement
st.info("""
**Thesis Objective:** Unlike existing static excipient-intake calculators, structural degradation-pathway predictors, and single-condition conservative kinetic models, this work develops an open-source, stochastic, time-dependent in-silico framework that integrates API structural vulnerability, excipient nitrite variability, multi-stress kinetic conditions, and CPCA-based acceptable-intake comparison to prioritize N-nitrosamine risk in solid oral dosage forms prior to confirmatory UPLC validation.
""")

if 'auto_ai_limit' not in st.session_state:
    st.session_state.auto_ai_limit = 1500.0  
if 'api_mw' not in st.session_state:
    st.session_state.api_mw = 300.0

tab1, tab2, tab3, tab4 = st.tabs(["🧬 API Structure", "💊 Formulation Matrix", "🔥 Stress Vectors", "📊 Kinetics & Risk Output"])

# --- TAB 1: 3D CHEMISTRY & STRUCTURE ---
with tab1:
    st.subheader("API Structural Vulnerability")
    
    try:
        with open("drug_database.json", "r") as f:
            portfolio = json.load(f)
    except FileNotFoundError:
        st.error("🛑 Database Error: 'drug_database.json' file not found.")
        portfolio = {"Fallback API (Paracetamol)": "CC(=O)NC1=CC=C(O)C=C1"}
    
    selected_drug = st.selectbox("Select API from Offline Database:", list(portfolio.keys()))
    final_smiles = portfolio[selected_drug]

    if final_smiles:
        molecule = Chem.MolFromSmiles(final_smiles)
        if molecule:
            formula = rdMolDescriptors.CalcMolFormula(molecule)
            mol_weight = Descriptors.MolWt(molecule)
            st.session_state.api_mw = mol_weight
            
            sec_amine_pattern = Chem.MolFromSmarts("[NX3;H1;!$(NC=O)]([C,c])([C,c])")
            tert_amine_pattern = Chem.MolFromSmarts("[NX3;H0;!$(NC=O)]([C,c])([C,c])([C,c])")
            
            has_sec = molecule.HasSubstructMatch(sec_amine_pattern)
            has_tert = molecule.HasSubstructMatch(tert_amine_pattern)
            
            st.info(f"**Formula:** {formula} | **Weight:** {mol_weight:.2f} g/mol")
            
            # Amine Speciation Parameter
            st.markdown("**Ionization Profile (Speciation)**")
            pKa = st.slider("API Amine pKa", 1.0, 14.0, 9.0, help="Used via Henderson-Hasselbalch to calculate free-base fraction available for nitrosation.")
            
            if has_sec:
                st.error("⚠️ **HIGH RISK:** Vulnerable Secondary (2°) Amine identified.")
                st.session_state.auto_ai_limit = 26.5
            elif has_tert:
                st.warning("⚠️ **MODERATE RISK:** Tertiary (3°) Amine identified.")
                st.session_state.auto_ai_limit = 400.0
            else:
                st.success("✅ **NO VULNERABLE AMINES DETECTED.**")
                st.session_state.auto_ai_limit = 1500.0
            
            molecule = Chem.AddHs(molecule)
            res = AllChem.EmbedMolecule(molecule, randomSeed=42)
            if res != 0: AllChem.Compute2DCoords(molecule)
            
            mblock = Chem.MolToMolBlock(molecule)
            view = py3Dmol.view(width=350, height=350)
            view.addModel(mblock, 'mol')
            view.setStyle({'stick': {}})
            view.zoomTo()
            components.html(view._make_html(), height=360, width=360)

# --- TAB 2: FORMULATION MATRIX ---
with tab2:
    st.subheader("API & Excipient Baseline Matrix")
    st.markdown("Calculate cumulative trace nitrites and define antioxidant/scavenger competition.")
    
    st.session_state.api_mass = st.number_input("API Mass per Unit (mg)", value=50.0, min_value=0.1)
    
    if 'formulation_df' not in st.session_state:
        st.session_state.formulation_df = pd.DataFrame({
            "Excipient": ["Microcrystalline Cellulose", "Sodium Starch Glycolate", "Ascorbic Acid (Scavenger)"],
            "Amount (mg)": [250.0, 30.0, 0.0],
            "Nitrite Mean (ppm)": [2.4, 1.5, 0.0],
            "Nitrite StdDev (ppm)": [1.1, 0.6, 0.0]
        })
    st.session_state.edited_df = st.data_editor(st.session_state.formulation_df, num_rows="dynamic", use_container_width=True)

# --- TAB 3: STRESS VECTORS ---
with tab3:
    st.subheader("Forced Degradation Testing Matrix")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🧪 Phase 1 (Acidic/Hydrolysis)")
        t1_c = st.slider("Stress Temperature (°C)", 40, 100, 80, key="t1_c")
        rh1 = st.slider("Chamber Humidity (%)", 0, 100, 75, key="rh1")
        time1 = st.number_input("Duration (Hours)", value=24.0, min_value=0.0, key="time1")
        micro_ph = st.slider("Formulation Microenvironmental pH", 1.0, 10.0, 3.0, help="Drives both N2O3 formation and amine protonation.")
        
    with col2:
        st.markdown("##### 🔥 Phase 2 (Thermal/Oxidative)")
        t2_c = st.slider("Stress Temperature (°C)", 40, 120, 100, key="t2_c")
        rh2 = st.slider("Chamber Humidity (%)", 0, 100, 20, key="rh2")
        time2 = st.number_input("Duration (Hours)", value=48.0, min_value=0.0, key="time2")

# --- TAB 4: KINETICS & RISK OUTPUT ---
with tab4:
    st.subheader("Stochastic ODE Degradation Solver")
    
    with st.expander("Methodology & Mathematical Assumptions", expanded=False):
        st.markdown("""
        * **Experimental Validation Required:** This tool prioritizes risk based on first-principles kinetics. Outputs must be verified via physical LC-MS/GC-MS stress testing.
        * **Input Distributions:** Nitrite Monte Carlo draws assume literature distributions from the Boetzel (2022) Excipient Database.
        * **Amine Speciation:** Relies on ideal Henderson-Hasselbalch behavior in solid-state microenvironments.
        """)

    max_dose = st.number_input("Max Daily Dose of API (mg/day)", value=150.0, min_value=0.1)
    ppm_limit = st.session_state.auto_ai_limit / max_dose
    st.info(f"**Target Detection Limit:** {ppm_limit:.4f} ppm (ng/mg API)")
    
    if st.button("🚀 Execute Risk Prioritization Engine", type="primary", use_container_width=True):
        
        df = st.session_state.edited_df
        
        with st.spinner("Integrating competing degradation pathways & speciation (5,000 draws)..."):
            rng = np.random.default_rng(42)
            mc_draws = 5000
            
            excipient_mass_mg = df["Amount (mg)"].sum()
            total_mass_mg = excipient_mass_mg + st.session_state.api_mass
            
            # Scavenger calculation
            scavenger_rows = df[df["Excipient"].str.contains("Scavenger|Ascorbic|Tocopherol", case=False, na=False)]
            scavenger_mass = scavenger_rows["Amount (mg)"].sum() if not scavenger_rows.empty else 0.0
            scavenge_multiplier = np.clip(scavenger_mass / 10.0, 0, 5.0) # Heuristic scavenger kinetics
            
            base_nitrite_ppm = np.zeros(mc_draws)
            for idx, row in df.iterrows():
                if row["Amount (mg)"] > 0:
                    fraction = row["Amount (mg)"] / total_mass_mg
                    samples = rng.normal(row["Nitrite Mean (ppm)"], row["Nitrite StdDev (ppm)"], mc_draws)
                    base_nitrite_ppm += np.clip(samples, 0, None) * fraction
            
            Ea_loss = rng.normal(75.0, 3.0, mc_draws) * 1000
            Ea_form = rng.normal(85.0, 4.0, mc_draws) * 1000
            f_max = np.clip(rng.normal(0.15, 0.03, mc_draws), 0, 1) 
            
            R = 8.314
            MW_nitrite = 46.0
            MW_NA = st.session_state.api_mw + 29.0 
            
            # Amine Speciation: Fraction of unprotonated amine (free base)
            # pH = pKa + log(Free / Protonated) -> Free fraction = 1 / (1 + 10^(pKa - pH))
            free_base_fraction = 1.0 / (1.0 + 10**(pKa - micro_ph))
            
            def calculate_rates(T_c, RH, Ea_l, Ea_f):
                T_k = T_c + 273.15
                k_loss = np.exp(21.0 - (Ea_l / (R * T_k)) + (0.02 * RH))
                
                # Formation requires N2O3 (favored at low pH) AND free amine
                acidic_enhancement = max(0, 5.0 - micro_ph) * 0.8
                k_form_base = np.exp(24.0 - (Ea_f / (R * T_k)) + (0.015 * RH) + acidic_enhancement)
                
                # Apply speciation limit
                k_form = k_form_base * free_base_fraction
                
                # Active Scavenging pathway
                k_scavenge = k_loss * scavenge_multiplier 
                
                return k_loss, k_form, k_scavenge
            
            final_NA_realistic = np.zeros(mc_draws)
            final_NA_conservative = np.zeros(mc_draws)
            
            for i in range(mc_draws):
                N0 = base_nitrite_ppm[i]
                cap = f_max[i] * N0 * (MW_NA / MW_nitrite) 
                
                k_l1, k_f1, k_s1 = calculate_rates(t1_c, rh1, Ea_loss[i], Ea_form[i])
                k_l2, k_f2, k_s2 = calculate_rates(t2_c, rh2, Ea_loss[i], Ea_form[i])
                
                # REALISTIC ODE (Includes active scavenging depletion)
                k_tot1 = k_l1 + k_f1 + k_s1
                N_p1 = N0 * np.exp(-k_tot1 * time1)
                NA_p1 = (k_f1 * N0 / k_tot1) * (1 - np.exp(-k_tot1 * time1)) if k_tot1 > 0 else 0
                
                k_tot2 = k_l2 + k_f2 + k_s2
                NA_final_real = NA_p1 + (k_f2 * N_p1 / k_tot2) * (1 - np.exp(-k_tot2 * time2)) if k_tot2 > 0 else NA_p1
                
                NA_final_real = NA_final_real * (MW_NA / MW_nitrite)
                final_NA_realistic[i] = min(NA_final_real, cap)
                
                # CONSERVATIVE ODE (Zero benign loss, zero scavenging)
                NA_cons = N0 * (1 - np.exp(-k_f1 * time1)) 
                NA_cons += (N0 - NA_cons) * (1 - np.exp(-k_f2 * time2))
                final_NA_conservative[i] = NA_cons * (MW_NA / MW_nitrite)
                
                if final_NA_conservative[i] < final_NA_realistic[i]:
                    final_NA_conservative[i] = final_NA_realistic[i]
                    
            dilution_factor = total_mass_mg / st.session_state.api_mass
            final_NA_realistic *= dilution_factor
            final_NA_conservative *= dilution_factor
                
            p95_real = np.percentile(final_NA_realistic, 95)
            p95_cons = np.percentile(final_NA_conservative, 95)
            
            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.metric("Conservative (P95 Worst-Case)", f"{p95_cons:.4f} ppm")
            col2.metric("Mechanistic Model (P95 Yield)", f"{p95_real:.4f} ppm")
            col3.metric("Regulatory Limit", f"{ppm_limit:.4f} ppm")
            
            if p95_cons > ppm_limit:
                st.error("🚨 **High Risk Prioritization:** Predicted yield exceeds acceptable intake. Confirmatory LC-MS/GC-MS testing is strictly required for this formulation under stress.")
            else:
                st.success("✅ **Low Risk Prioritization:** Formulation demonstrates robust kinetic resistance to nitrosation under requested stress bounds.")
                
