"""Government Scheme Matching Tools for Startup India Advisor

These tools provide:
- Central and state scheme matching based on business profile
- Subsidy value estimation
- Application guidance
"""

import json
import os
from typing import Dict, List

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

SCHEMES = {}
try:
    with open(os.path.join(_DATA_DIR, 'schemes.json')) as f:
        SCHEMES = json.load(f)
except FileNotFoundError:
    pass


def match_government_schemes(
    business_type: str,
    sector: str,
    investment_size_inr: float,
    state: str,
    employment_target: int = 0,
    is_startup: bool = False,
    is_msme: bool = False
) -> Dict:
    """Match business profile with applicable government schemes.

    Args:
        business_type: 'manufacturing', 'service', 'trading', 'tech'
        sector: Specific sector (e.g., 'electronics', 'textile', 'IT')
        investment_size_inr: Planned investment in INR
        state: Target state
        employment_target: Expected employment generation
        is_startup: Whether DPIIT-recognized startup
        is_msme: Whether MSME registered

    Returns:
        Matched central and state schemes with benefit estimates
    """
    matched = {"central": [], "state": []}

    # Match central schemes
    for scheme in SCHEMES.get("central", []):
        score = 0
        reasons = []

        # Sector match (30 points)
        scheme_sectors = [s.lower() for s in scheme.get("sector", [])]
        if sector.lower() in scheme_sectors or "all" in scheme_sectors:
            score += 30
            reasons.append("Sector match")

        # Investment threshold (25 points)
        min_inv = scheme.get("min_investment_cr", 0) * 10000000
        if investment_size_inr >= min_inv:
            score += 25
            reasons.append("Investment threshold met")

        # Startup/MSME eligibility (20 points each)
        if is_startup and "dpiit" in str(scheme.get("eligibility", [])).lower():
            score += 20
            reasons.append("Startup eligible")
        if is_msme and "msme" in str(scheme.get("eligibility", [])).lower():
            score += 20
            reasons.append("MSME eligible")

        # Employment target bonus (5 points)
        if employment_target > 0 and employment_target >= 50:
            score += 5
            reasons.append("Employment target met")

        if score >= 40:
            matched["central"].append({
                "scheme": scheme["name"],
                "benefit": scheme["benefit"],
                "match_score": min(score, 100),
                "reasons": reasons,
                "url": scheme.get("url", ""),
                "min_investment_cr": scheme.get("min_investment_cr", 0)
            })

    # Match state schemes
    state_key = state.lower().replace(" ", "_")
    state_schemes = SCHEMES.get("state", {}).get(state_key, [])

    for scheme in state_schemes:
        scheme_sectors = scheme.get("sector", [])
        sector_match = (
            scheme_sectors == "all" or 
            sector.lower() in [s.lower() for s in scheme_sectors]
        )

        if sector_match:
            matched["state"].append({
                "scheme": scheme["name"],
                "benefit": scheme["benefit"],
                "match_score": 85,
                "reasons": ["State-specific scheme", "Sector match"]
            })

    # Sort by match score descending
    matched["central"].sort(key=lambda x: x["match_score"], reverse=True)

    return {
        "status": "success",
        "total_schemes": len(matched["central"]) + len(matched["state"]),
        "central_count": len(matched["central"]),
        "state_count": len(matched["state"]),
        "schemes": matched
    }


def estimate_subsidy_value(
    scheme_name: str,
    investment_size_inr: float,
    projected_revenue_inr: float = 0
) -> Dict:
    """Estimate actual monetary value of a subsidy/scheme.

    Args:
        scheme_name: Name of the scheme
        investment_size_inr: Total investment
        projected_revenue_inr: Annual projected revenue (optional)

    Returns:
        Estimated subsidy amount and timeline
    """
    # Simplified estimation logic based on scheme type
    estimates = {
        "PLI Scheme - Electronics": min(projected_revenue_inr * 0.05, investment_size_inr * 0.3),
        "PLI Scheme - Automobile": min(projected_revenue_inr * 0.12, investment_size_inr * 0.4),
        "PLI Scheme - Textiles": min(projected_revenue_inr * 0.08, investment_size_inr * 0.25),
        "Startup India Seed Fund Scheme (SISFS)": min(50000000, investment_size_inr * 0.2),
        "MSME Credit Guarantee Fund Trust (CGTMSE)": min(50000000, investment_size_inr * 0.8),
        "MUDRA Loan (Pradhan Mantri Mudra Yojana)": min(1000000, investment_size_inr * 0.1),
        "Stand-Up India Scheme": min(100000000, investment_size_inr * 0.5),
        "SIDBI Fund of Funds for Startups (FFS)": min(25000000, investment_size_inr * 0.15),
        "Gujarat Industrial Policy 2020": investment_size_inr * 0.10,
        "Gujarat EV Policy 2021": investment_size_inr * 0.15,
        "Dholera SIR Incentives": investment_size_inr * 0.20,
        "Karnataka Startup Policy 2022": 5000000,
        "Karnataka EV Policy": investment_size_inr * 0.15,
        "Karnataka Semiconductor Policy": investment_size_inr * 0.25,
        "Tamil Nadu Industrial Policy 2021": investment_size_inr * 0.08,
        "TN EV Policy 2023": investment_size_inr * 0.15,
        "TN Startup & Innovation Policy": 10000000,
        "Magnetic Maharashtra 2.0": investment_size_inr * 0.08,
        "Maharashtra EV Policy": investment_size_inr * 0.15,
        "Mumbai Fintech Sandbox": 2500000,
        "TS-iPASS (Telangana State Industrial Project Approval and Self-Certification System)": 0,
        "Telangana EV Policy": investment_size_inr * 0.10,
        "T-Hub Startup Support": 2500000,
        "UP Electronics Manufacturing Policy": investment_size_inr * 0.25,
        "UP EV Manufacturing Policy": investment_size_inr * 0.20,
        "ODOP (One District One Product) Scheme": 2500000
    }

    estimated = estimates.get(scheme_name, investment_size_inr * 0.05)

    # Determine timeline based on scheme type
    if "PLI" in scheme_name:
        timeline = "5 years (annual disbursement)"
    elif "Startup" in scheme_name or "Seed" in scheme_name:
        timeline = "3-6 months"
    elif "Credit" in scheme_name or "MUDRA" in scheme_name or "Loan" in scheme_name:
        timeline = "1-3 months"
    elif "EV" in scheme_name:
        timeline = "2-4 years (phased)"
    else:
        timeline = "6-18 months"

    return {
        "status": "success",
        "scheme": scheme_name,
        "estimated_value_inr": round(estimated, 0),
        "estimated_value_crores": round(estimated / 10000000, 2),
        "timeline": timeline,
        "disbursement_type": "Reimbursement / Direct Transfer / Subsidy"
    }
