import streamlit as st
import pandas as pd
import numpy as np
import datetime
import hashlib
import json
import requests
import urllib.parse
import re
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
import py3Dmol
import streamlit.components.v1 as components

APP_VERSION = "0.5.0-prototype"
MC_SEED = 42
MC_DRAWS = 5000

# ---------------------------------------------------------------------------
# SYNTHETIC kinetic parameters. NOT FOR REGULATORY USE.
# Replace with a registry that carries provenance (LIMS ID / DOI / consortium ID),
# units, uncertainty and an applicability domain before any compliance decision.
# ---------------------------------------------------------------------------
SYNTH = {
    "provenance": "SYNTHETIC - NOT FOR REGULATORY USE",
    "ea_loss_kj_mean_sd": (75.0, 3.0),
    "ea_form_kj_mean_sd": (85.0, 4.0),
    "f_max_mean_sd": (0.12, 0.02),      # realistic-mode conversion cap (molar fraction)
    "f_max_upper_bound": 0.16,          # conservative-mode conversion cap (molar fraction)
}

st.set_page_config(page_title="NitroPredict Engine", layout="wide")
st.title("NitroPredict: Stochastic QbD Assessment Engine")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🧬 Structure", "💊 Ledger", "⚖️ Limits", "⚗️ Kinetics Timeline", "📊 Report"]
)

# --- Session defaults (set BEFORE any widget that uses these keys) ---
if "auto_ai_limit" not in st.session_state:
    st.session_state.auto_ai_limit = 1500.0
if "ai_limit" not in st.session_state:
    st.session_state.ai_limit = st.session_state.auto_ai_limit
if "api_mw" not in st.session_state:
    st.session_state.api_mw = 300.0
if "amine_class" not in st.session_state:
    st.session_state.amine_class = "none"

# --- TAB 1: STRUCTURE VIEWER & AMINE SCREEN ---
structure_ok = False
final_smiles = ""

with tab1:
    st.subheader("Structure Viewer & Amine-Class Screen")

    st.write("Select API from database or search live:")
    portfolio = {
        "Custom Search (Live PubChem API)": "",
        "Paracetamol (Amide - no vulnerable amine)": "CC(=O)NC1=CC=C(O)C=C1",
        "Metformin (N,N-dimethylguanidine)": "CN(C)C(=NC(=N)N)N",
        "Valsartan (Amide - no vulnerable amine)": "CCCCC(=O)N(CC1=CC=C(C=C1)C2=CC=CC=C2C3=NNN=N3)C(C(C)C)C(=O)O",
    }
    selected_drug = st.selectbox("Select API:", list(portfolio.keys()), label_visibility="collapsed")

    if selected_drug == "Custom Search (Live PubChem API)":
        search_name = st.text_input("Enter API Name to fetch from NIH PubChem:")
        if search_name:
            encoded_name = urllib.parse.quote(search_name.strip())
            with st.spinner(f"Querying NIH PubChem for '{search_name}'..."):
                try:
                    url = (
                        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                        f"{encoded_name}/property/CanonicalSMILES/JSON"
                    )
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        final_smiles = response.json()["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
                        st.success("✅ Live Connection Successful.")
                    else:
                        st.error("⚠️ Drug not found in PubChem.")
                except Exception:
                    st.error("Error connecting to external database.")
    else:
        final_smiles = portfolio[selected_drug]

    if final_smiles:
        molecule = Chem.MolFromSmiles(final_smiles)
        if molecule is not None and molecule.GetNumAtoms() > 0:
            structure_ok = True
            formula = rdMolDescriptors.CalcMolFormula(molecule)
            mol_weight = Descriptors.MolWt(molecule)
            st.session_state.api_mw = mol_weight

            # Amine-class screen (amides, ureas and carbamates excluded). This is NOT CPCA scoring.
            sec_amine_pattern = Chem.MolFromSmarts("[NX3;H1;!$(NC=O)]([C,c])([C,c])")
            tert_amine_pattern = Chem.MolFromSmarts("[NX3;H0;!$(NC=O)]([C,c])([C,c])([C,c])")
            has_sec = molecule.HasSubstructMatch(sec_amine_pattern)
            has_tert = molecule.HasSubstructMatch(tert_amine_pattern)

            st.info(f"**Formula:** {formula} | **Weight:** {mol_weight:.2f} g/mol")

            if has_sec:
                amine_class, new_limit = "secondary", 26.5
                st.error("⚠️ **HIGH RISK:** Vulnerable secondary (2°) amine identified (amides excluded).")
            elif has_tert:
                amine_class, new_limit = "tertiary", 400.0
                st.warning("⚠️ **MODERATE RISK:** Tertiary (3°) amine identified.")
            else:
                amine_class, new_limit = "none", 1500.0
                st.success(
                    "✅ **No vulnerable 2°/3° amine in the API structure.** "
                    "Check related-substance and residual amines (e.g. dimethylamine) separately."
                )
            st.session_state.amine_class = amine_class

            # Only reset the limit when the structure CHANGES, so a manual override survives reruns.
            if st.session_state.get("last_smiles") != final_smiles:
                st.session_state.auto_ai_limit = new_limit
                st.session_state.ai_limit = new_limit
                st.session_state.last_smiles = final_smiles

            st.caption(
                "Placeholder limits by amine class (26.5 / 400 / 1500 ng/day), not CPCA scoring. "
                "Set the AI for your product from current FDA/EMA guidance in the Limits tab."
            )

            # 3D conformer
            mol3d = Chem.AddHs(molecule)
            if AllChem.EmbedMolecule(mol3d, randomSeed=42) != 0:
                AllChem.Compute2DCoords(mol3d)
                st.caption("3D embedding failed; showing a flat 2D layout.")
            mblock = Chem.MolToMolBlock(mol3d)
            view = py3Dmol.view(width=350, height=350)
            view.addModel(mblock, "mol")
            view.setStyle({"stick": {}})
            view.zoomTo()
            html = view.write_html() if hasattr(view, "write_html") else view._make_html()
            components.html(html, height=360, width=360)
        else:
            st.error("Invalid structure. Simulation is blocked until a valid structure is selected.")

# --- TAB 2: SUPPLY CHAIN (Stochastic Ledger) ---
with tab2:
    st.subheader("Stochastic Supply Chain Ledger")

    api_mass_per_pill = st.number_input("API Mass per Pill (mg)", value=500.0, min_value=0.1)
    st.session_state.api_mass = api_mass_per_pill

    if "formulation_df" not in st.session_state:
        st.session_state.formulation_df = pd.DataFrame({
            "Excipient": ["Microcrystalline Cellulose", "Magnesium Stearate", "Sodium Starch Glycolate"],
            "Vendor": ["FMC Biopolymer", "Mallinckrodt", "Roquette"],
            "Amount (mg/pill)": [100.0, 5.0, 20.0],
            "Nitrite Mean (ppm)": [2.5, 1.0, 0.8],
            "Nitrite StdDev (ppm)": [0.5, 0.2, 0.1],
        })
    st.session_state.edited_df = st.data_editor(
        st.session_state.formulation_df, num_rows="dynamic", use_container_width=True
    )

# --- TAB 3: REGULATORY CALCULATOR ---
with tab3:
    st.subheader("Acceptable Intake (AI) Calculator")
    # No value= here: the initial value comes from session state, so there is no default/state conflict.
    ai_limit = st.number_input("Acceptable Intake (ng/day)", min_value=0.1, key="ai_limit")
    max_dose = st.number_input("Max Daily Dose of API (mg/day)", value=500.0, key="max_dose", min_value=0.1)

    ppm_limit = ai_limit / max_dose
    st.success(f"**Actionable Concentration Limit:** {ppm_limit:.4f} ppm (ng of NA / mg of API)")

# --- TAB 4: KINETIC TIMELINE & REGISTRY ---
with tab4:
    st.subheader("Piecewise Environmental Timeline & Parameters")
    is_validated = st.checkbox(
        "I confirm these kinetic parameters are experimentally validated (self-attestation).",
        key="kinetics_validated",
    )

    if is_validated:
        st.selectbox(
            "Select Documentation Source Type:",
            ["LIMS / AR&D Report", "DOI (Literature)", "Consortium ID"],
            key="source_type",
        )
        st.text_input("Enter Official Document ID:", key="validation_source")
        st.warning(
            "The built-in kinetic parameters are SYNTHETIC. A document ID you enter here does not make them validated."
        )

    with st.expander("🏭 Phase 1: Manufacturing / Accelerated", expanded=True):
        t1_c = st.slider("Phase 1 Temperature (°C)", 20, 80, 60, key="t1_c")
        rh1 = st.slider("Phase 1 Internal RH (%)", 0, 100, 40, key="rh1")
        time1 = st.number_input("Phase 1 Duration (Days)", value=2.0, key="time1", min_value=0.0)

    with st.expander("📦 Phase 2: Long-Term Storage", expanded=True):
        t2_c = st.slider("Phase 2 Temperature (°C)", 0, 60, 25, key="t2_c")
        rh2 = st.slider("Phase 2 Internal RH (%)", 0, 100, 60, key="rh2")
        time2 = st.number_input("Phase 2 Duration (Days)", value=730.0, key="time2", min_value=0.0)

    ae_pct = st.number_input("Alkaline Excipient Mass (%)", value=5.0, key="ae_pct", min_value=0.0)

    st.caption("Note: kinetic parameters (Ea, f_max) are synthetic prototypes pending lab integration.")

# --- TAB 5: MASTER ENGINE (analytic two-phase solution + Monte Carlo) ---
with tab5:
    st.subheader("Stochastic Kinetic Assessment")

    if st.button("🚀 Execute Monte Carlo Engine", type="primary", use_container_width=True):

        # ---------------- PRE-FLIGHT CHECKS ----------------
        if not structure_ok:
            st.error("🛑 Blocked: a valid API structure is required.")
            st.stop()

        num_cols = ["Amount (mg/pill)", "Nitrite Mean (ppm)", "Nitrite StdDev (ppm)"]
        df = st.session_state.edited_df.dropna(how="all").copy()
        if df.empty:
            st.error("🛑 Blocked: formulation ledger is empty.")
            st.stop()
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
        if df[num_cols].isna().any().any():
            st.error("🛑 Blocked: blank or non-numeric values in the ledger.")
            st.stop()
        if (df[num_cols] < 0).any().any():
            st.error("🛑 Blocked: negative amounts, nitrite means or standard deviations are not allowed.")
            st.stop()
        if df["Amount (mg/pill)"].sum() <= 0:
            st.error("🛑 Blocked: ledger has zero excipient mass.")
            st.stop()

        validated = bool(st.session_state.get("kinetics_validated", False))
        source_type = st.session_state.get("source_type", "") if validated else ""
        doc_id = st.session_state.get("validation_source", "").strip() if validated else ""

        if validated:
            if not doc_id:
                st.error("🛑 DATA INTEGRITY LOCK: Document ID missing. Traceability required for a validated run.")
                st.stop()
            if "DOI" in source_type:
                if not re.search(r"^10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+$", doc_id):
                    st.error("🛑 DATA INTEGRITY LOCK: Invalid DOI format.")
                    st.stop()
            else:
                # Generic shape check only. Replace with your organisation's real LIMS / consortium ID formats.
                if not re.search(r"^[A-Za-z0-9][A-Za-z0-9._/\-]{3,63}$", doc_id):
                    st.error("🛑 DATA INTEGRITY LOCK: Document ID format not recognised (4-64 characters: letters, digits, . _ / -).")
                    st.stop()

        # ---------------- SIMULATION ----------------
        with st.spinner(f"Evaluating the coupled rate equations across {MC_DRAWS:,} Monte Carlo draws..."):

            rng = np.random.default_rng(MC_SEED)

            api_mass = st.session_state.api_mass
            total_mass_mg = df["Amount (mg/pill)"].sum() + api_mass

            base_nitrite_ppm = np.zeros(MC_DRAWS)   # ppm of the WHOLE tablet
            for _, row in df.iterrows():
                if row["Amount (mg/pill)"] > 0:
                    fraction = row["Amount (mg/pill)"] / total_mass_mg
                    samples = rng.normal(row["Nitrite Mean (ppm)"], row["Nitrite StdDev (ppm)"], MC_DRAWS)
                    base_nitrite_ppm += np.clip(samples, 0, None) * fraction

            Ea_loss = rng.normal(*SYNTH["ea_loss_kj_mean_sd"], MC_DRAWS) * 1000
            Ea_form = rng.normal(*SYNTH["ea_form_kj_mean_sd"], MC_DRAWS) * 1000
            f_max = np.clip(rng.normal(*SYNTH["f_max_mean_sd"], MC_DRAWS), 0, 1)
            # Conservative cap is never below the realistic cap of the same draw, so ordering is guaranteed.
            f_cons = np.maximum(f_max, SYNTH["f_max_upper_bound"])

            R = 8.314
            MW_nitrite = 46.0
            MW_NA = st.session_state.api_mw + 29.0
            ratio = MW_NA / MW_nitrite

            def calculate_rates(T_c, RH, Ea_l, Ea_f):
                T_k = T_c + 273.15
                k_loss = np.exp(25.0 - (Ea_l / (R * T_k)) + (0.02 * RH))
                k_form = np.exp(28.0 - (Ea_f / (R * T_k)) + (0.015 * RH) - (0.1 * ae_pct))
                return k_loss, k_form

            final_NA_realistic = np.zeros(MC_DRAWS)
            final_NA_conservative = np.zeros(MC_DRAWS)
            t1, t2 = st.session_state.time1, st.session_state.time2

            for i in range(MC_DRAWS):
                N0 = base_nitrite_ppm[i]

                k_l1, k_f1 = calculate_rates(st.session_state.t1_c, st.session_state.rh1, Ea_loss[i], Ea_form[i])
                k_l2, k_f2 = calculate_rates(st.session_state.t2_c, st.session_state.rh2, Ea_loss[i], Ea_form[i])

                # REALISTIC (nitrite loss + conversion cap)
                k_tot1 = k_l1 + k_f1
                N_p1 = N0 * np.exp(-k_tot1 * t1)
                NA_p1 = (k_f1 * N0 / k_tot1) * (1 - np.exp(-k_tot1 * t1))
                k_tot2 = k_l2 + k_f2
                NA_real = NA_p1 + (k_f2 * N_p1 / k_tot2) * (1 - np.exp(-k_tot2 * t2))
                final_NA_realistic[i] = min(NA_real * ratio, f_max[i] * N0 * ratio)

                # CONSERVATIVE (no nitrite loss, capped at the upper-bound conversion)
                NA_cons = N0 * (1 - np.exp(-k_f1 * t1))
                NA_cons += (N0 - NA_cons) * (1 - np.exp(-k_f2 * t2))
                final_NA_conservative[i] = min(NA_cons * ratio, f_cons[i] * N0 * ratio)

            # Model-consistency check: conservative must not fall below realistic. Fail loudly, never patch.
            violations = int(np.sum(final_NA_conservative < final_NA_realistic - 1e-12))
            if violations > 0:
                st.error(
                    f"🛑 Model consistency error: conservative < realistic in {violations} draws. "
                    "Results withheld. Report this as a defect."
                )
                st.stop()

            # Convert 'ppm of whole tablet' to 'ppm of API' for direct comparison with the limit
            dilution_factor = total_mass_mg / api_mass
            final_NA_realistic = final_NA_realistic * dilution_factor
            final_NA_conservative = final_NA_conservative * dilution_factor

            pr = np.percentile(final_NA_realistic, [50, 95, 99])
            pc = np.percentile(final_NA_conservative, [50, 95, 99])
            p95_real, p95_cons = pr[1], pc[1]
            ppm_limit = st.session_state.ai_limit / st.session_state.max_dose

            if not (np.all(np.isfinite(pr)) and np.all(np.isfinite(pc))):
                st.error("🛑 Math error: non-finite result. Check inputs for blank values.")
                st.stop()

        # ---------------- RESULTS ----------------
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Conservative (P95)", f"{p95_cons:.4f} ppm")
        col2.metric("Realistic Mode (P95)", f"{p95_real:.4f} ppm")
        col3.metric("Regulatory Limit", f"{ppm_limit:.4f} ppm")
        st.caption(
            f"Realistic P50 / P99: {pr[0]:.4f} / {pr[2]:.4f} ppm | "
            f"Conservative P50 / P99: {pc[0]:.4f} / {pc[2]:.4f} ppm (ng NA per mg API)"
        )

        decision = "NOT ISSUED"
        if not validated:
            st.warning("⚠️ **INDICATIVE ONLY:** run is not self-attested as validated. No compliance decision issued.")
        if st.session_state.amine_class == "none":
            st.info(
                "ℹ️ No vulnerable amine in the API structure, so no API-derived nitrosamine pathway is modelled "
                "and no PASS/FAIL is issued. Document this conclusion, and assess amine impurities separately."
            )
        elif validated:
            st.info(
                f"🔒 **Self-attested run record (parameters synthetic; not saved on the server):** "
                f"{source_type} | ID: `{doc_id}`"
            )
            if p95_cons > ppm_limit:
                decision = "FAIL"
                st.error("❌ **FAIL:** at P95, the conservative case exceeds the acceptable limit.")
            else:
                decision = "PASS"
                st.success("✅ **PASS:** the conservative P95 remains below the limit.")

        # ---------------- RUN RECORD (unsigned, downloadable) ----------------
        record = {
            "app_version": APP_VERSION,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "structure": {
                "smiles": final_smiles,
                "api_mw": float(st.session_state.api_mw),
                "amine_class": st.session_state.amine_class,
            },
            "limits": {
                "ai_ng_per_day": float(st.session_state.ai_limit),
                "max_daily_dose_mg": float(st.session_state.max_dose),
                "limit_ppm_of_api": float(ppm_limit),
            },
            "api_mass_per_pill_mg": float(api_mass),
            "ledger": df.to_dict("records"),
            "timeline": {
                "phase1": {"T_c": st.session_state.t1_c, "RH": st.session_state.rh1, "days": float(t1)},
                "phase2": {"T_c": st.session_state.t2_c, "RH": st.session_state.rh2, "days": float(t2)},
                "alkaline_excipient_pct": float(ae_pct),
            },
            "parameters": SYNTH,
            "monte_carlo": {"seed": MC_SEED, "draws": MC_DRAWS},
            "attestation": {"self_attested_validated": validated, "source_type": source_type, "document_id": doc_id},
            "results_ppm_of_api": {
                "realistic_p50_p95_p99": [float(x) for x in pr],
                "conservative_p50_p95_p99": [float(x) for x in pc],
            },
            "decision": decision,
        }
        payload = json.dumps(record, sort_keys=True, indent=2, default=str)
        record_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        st.download_button(
            "⬇️ Download run record (JSON)",
            data=payload,
            file_name=f"nitropredict_run_{record_hash[:12]}.json",
            mime="application/json",
        )
        st.caption(f"SHA-256 of record: `{record_hash}`. Unsigned and not stored on the server.")
        
