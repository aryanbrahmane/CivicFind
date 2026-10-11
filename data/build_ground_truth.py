import json
import os

with open(os.path.join(os.path.dirname(__file__), "schemes.json"), "r", encoding="utf-8") as f:
    schemes = json.load(f)

# Map schemes by ID and name for easy lookup
scheme_map = {s["id"]: s for s in schemes}

BENCHMARK_QUERIES = [
    {
        "query_id": "Q01",
        "query": "scholarships for engineering college students in maharashtra",
        "category": "Education",
        "intent": "Find higher education scholarships and fee reimbursements for engineering and technical students in Maharashtra.",
        "relevant_docs": ["SCH-EDU-001", "SCH-EDU-002", "SCH-EDU-003", "SCH-EDU-004", "SCH-EDU-005", "SCH-EDU-009"],
        "graded_relevance": {
            "SCH-EDU-001": 2,  # Rajarshi Shahu Maharaj fee waiver - Direct match
            "SCH-EDU-005": 2,  # Panjabrao Deshmukh hostel allowance for engineering
            "SCH-EDU-004": 2,  # AICTE Pragati engineering scholarship
            "SCH-EDU-002": 1,  # Post-matric SC scholarship
            "SCH-EDU-003": 1,  # Central sector college scholarship
            "SCH-EDU-009": 1   # Minority professional course scholarship
        }
    },
    {
        "query_id": "Q02",
        "query": "financial assistance subsidies for small farmers in maharashtra",
        "category": "Agriculture",
        "intent": "Find direct income support, financial grants, and input subsidies for farmers in Maharashtra.",
        "relevant_docs": ["SCH-AGR-001", "SCH-AGR-002", "SCH-AGR-003", "SCH-AGR-006", "SCH-AGR-007", "SCH-AGR-010"],
        "graded_relevance": {
            "SCH-AGR-001": 2,  # PM KISAN direct income
            "SCH-AGR-002": 2,  # Namo Shetkari 6000 topup
            "SCH-AGR-003": 2,  # PMFBY 1-rupee crop insurance
            "SCH-AGR-006": 1,  # Ambedkar Krushi Swavalamban
            "SCH-AGR-007": 1,  # PoCRA climate resilient subsidy
            "SCH-AGR-010": 1   # KCC subsidized crop credit
        }
    },
    {
        "id": "Q03",
        "query_id": "Q03",
        "query": "cashless hospital surgery health insurance schemes",
        "category": "Healthcare",
        "intent": "Find government medical insurance providing free/cashless hospital admission and surgeries.",
        "relevant_docs": ["SCH-HLT-001", "SCH-HLT-002", "SCH-HLT-006", "SCH-HLT-009", "SCH-HLT-016"],
        "graded_relevance": {
            "SCH-HLT-001": 2,  # MJPJAY 5 lakh cashless cover in Maharashtra
            "SCH-HLT-002": 2,  # Ayushman Bharat PM-JAY 5 lakh cover
            "SCH-HLT-006": 2,  # CM Medical Relief Fund for major surgeries
            "SCH-HLT-009": 1,  # Rashtriya Arogya Nidhi BPL hospital grant
            "SCH-HLT-016": 1   # Balasaheb Thackeray emergency accident trauma aid
        }
    },
    {
        "query_id": "Q04",
        "query": "financial aid monthly cash assistance for women in maharashtra",
        "category": "Women and Child Welfare",
        "intent": "Discover monthly DBT cash transfer schemes empowering women financially.",
        "relevant_docs": ["SCH-WOM-001", "SCH-WOM-004", "SCH-WOM-006", "SCH-WOM-007", "SCH-SOC-001"],
        "graded_relevance": {
            "SCH-WOM-001": 2,  # Ladki Bahin 1500 monthly DBT
            "SCH-WOM-004": 2,  # Annapurna 3 free LPG cylinders
            "SCH-SOC-001": 1,  # Sanjay Gandhi Niradhar widow/destitute pension
            "SCH-WOM-006": 1,  # Navtejaswini rural SHG enterprise
            "SCH-WOM-007": 1   # Mahila Samman savings
        }
    },
    {
        "query_id": "Q05",
        "query": "unemployed youth skill development apprenticeship training and stipend",
        "category": "Employment and Skill Development",
        "intent": "Find job training, industrial apprenticeship, and stipend programs for educated youth.",
        "relevant_docs": ["SCH-EMP-001", "SCH-EMP-002", "SCH-EMP-003", "SCH-EMP-005", "SCH-EMP-009", "SCH-EMP-007"],
        "graded_relevance": {
            "SCH-EMP-001": 2,  # Yuva Karya Prashikshan up to 10000 stipend
            "SCH-EMP-005": 2,  # NAPS apprenticeship stipend
            "SCH-EMP-009": 2,  # PM Internship top 500 companies 5000 stipend
            "SCH-EMP-002": 1,  # PMKVY free skill certification
            "SCH-EMP-003": 1,  # CMEGP self employment subsidy
            "SCH-EMP-007": 1   # DDU-GKY residential training
        }
    },
    {
        "query_id": "Q06",
        "query": "pucca housing assistance subsidy for low income urban rural poor",
        "category": "Housing and Urban Development",
        "intent": "Discover government housing schemes providing grants or interest subsidies for constructing pucca houses.",
        "relevant_docs": ["SCH-HOU-001", "SCH-HOU-002", "SCH-HOU-003", "SCH-HOU-004", "SCH-HOU-006"],
        "graded_relevance": {
            "SCH-HOU-001": 2,  # PMAY-Urban 2.0
            "SCH-HOU-002": 2,  # PMAY-Gramin 1.2 lakh grant
            "SCH-HOU-003": 2,  # Ramai Awas Gharkul
            "SCH-HOU-004": 1,  # Shabari Awas ST Gharkul
            "SCH-HOU-006": 1   # Modi Awas OBC Gharkul
        }
    },
    {
        "query_id": "Q07",
        "query": "collateral free business loans for micro small enterprises and shopkeepers",
        "category": "Entrepreneurship and MSME",
        "intent": "Find collateral-free bank loans and credit guarantee programs for micro businesses.",
        "relevant_docs": ["SCH-ENT-001", "SCH-ENT-002", "SCH-ENT-003", "SCH-HOU-005", "SCH-EMP-004"],
        "graded_relevance": {
            "SCH-ENT-001": 2,  # PM Mudra Yojana up to 20 lakh
            "SCH-ENT-003": 2,  # CGTMSE collateral-free up to 5 crore
            "SCH-ENT-002": 2,  # PM Vishwakarma 3 lakh concessional loan
            "SCH-HOU-005": 1,  # PM SVANidhi street vendors
            "SCH-EMP-004": 1   # PMEGP capital subsidy loan
        }
    },
    {
        "query_id": "Q08",
        "query": "monthly pension financial support for senior citizens aged 65 and above",
        "category": "Social Security and Pensions",
        "intent": "Find old age pension and regular monthly financial aid for senior citizens.",
        "relevant_docs": ["SCH-SOC-002", "SCH-SOC-003", "SCH-SOC-001", "SCH-SOC-004", "SCH-SOC-008"],
        "graded_relevance": {
            "SCH-SOC-002": 2,  # Shravanbal 1500 monthly pension for 65+
            "SCH-SOC-003": 2,  # IGNOAPS national old age pension
            "SCH-SOC-001": 1,  # Sanjay Gandhi Niradhar destitute pension
            "SCH-SOC-004": 1,  # Atal Pension Yojana
            "SCH-SOC-008": 1   # PM Vaya Vandana Yojana LIC pension
        }
    },
    {
        "query_id": "Q09",
        "query": "assistive devices motorized tricycle and support for disabled persons",
        "category": "Disability Welfare",
        "intent": "Find physical aids, appliances, wheelchairs, and grants for differently-abled citizens.",
        "relevant_docs": ["SCH-DIS-001", "SCH-DIS-002", "SCH-DIS-003", "SCH-DIS-005", "SCH-SOC-005"],
        "graded_relevance": {
            "SCH-DIS-001": 2,  # ADIP scheme motorized tricycles and wheelchairs
            "SCH-DIS-002": 2,  # Maharashtra Divyang Swayamrojgar business loan
            "SCH-SOC-005": 1,  # Rashtriya Vayoshri assisted living devices
            "SCH-DIS-003": 1,  # National fellowship for PwDs
            "SCH-DIS-005": 1   # Accessible India Sugamya Bharat
        }
    },
    {
        "query_id": "Q10",
        "query": "rooftop solar panel subsidy free electricity for residential homes",
        "category": "Public Services and Citizen Welfare",
        "intent": "Find government capital subsidies for installing rooftop solar systems and zero electricity bills.",
        "relevant_docs": ["SCH-PUB-001", "SCH-AGR-004", "SCH-AGR-005", "SCH-PUB-008"],
        "graded_relevance": {
            "SCH-PUB-001": 2,  # PM Surya Ghar 78000 subsidy & 300 free units
            "SCH-AGR-004": 1,  # MSKVY 2.0 solar agricultural feeders
            "SCH-AGR-005": 1,  # PM KUSUM solar pump subsidy
            "SCH-PUB-008": 1   # Mahavitaran krishi pump electricity bill waiver
        }
    },
    {
        "query_id": "Q11",
        "query": "maternity benefit nutritious diet grant for pregnant women and lactating mothers",
        "category": "Healthcare",
        "intent": "Find cash incentives and nutrition support during pregnancy and child birth.",
        "relevant_docs": ["SCH-HLT-003", "SCH-HLT-014", "SCH-HLT-020", "SCH-WOM-008"],
        "graded_relevance": {
            "SCH-HLT-003": 2,  # PMMVY 5000 / 6000 direct cash incentive
            "SCH-HLT-014": 2,  # Janani Suraksha institutional delivery grant
            "SCH-WOM-008": 1,  # PMSMA monthly free antenatal checkups
            "SCH-HLT-020": 1   # Anemia Mukt Bharat iron & calcium supplementation
        }
    },
    {
        "query_id": "Q12",
        "query": "education loan subsidy interest waiver for higher studies",
        "category": "Education",
        "intent": "Find interest subvention and educational loan assistance portals.",
        "relevant_docs": ["SCH-EDU-014", "SCH-EDU-001", "SCH-EDU-003", "SCH-EDU-018"],
        "graded_relevance": {
            "SCH-EDU-014": 2,  # Vidya Lakshmi single window education loan & interest subsidy
            "SCH-EDU-018": 1,  # PM-USP higher education interest subsidy
            "SCH-EDU-001": 1,  # Rajarshi Shahu fee concession
            "SCH-EDU-003": 1   # Central Sector scholarship
        }
    }
]

output_path = os.path.join(os.path.dirname(__file__), "ground_truth.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(BENCHMARK_QUERIES, f, indent=2, ensure_ascii=False)

print(f"Successfully generated {len(BENCHMARK_QUERIES)} ground truth benchmark queries at {output_path}!")
