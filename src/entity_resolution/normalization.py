"""
Multi-View Text Normalization Pipeline
Amazon ML Challenge 2026 - Business Entity Resolution

Generates standardized, structured views for reference and target records:
- Transliteration of Indic scripts and European diacritics via anyascii
- Alphanumeric cleaning and whitespace normalization
- Legal entity suffix extraction (US, India, France)
- Structured address component extraction (house number, postal code, numeric tokens)
- Streaming batch processing to Parquet for high-throughput disk caching
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import re
import anyascii
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Legal suffix vocabulary (ordered by descending token length for greedy matching)
MULTI_TOKEN_SUFFIXES = [
    ("private", "limited"),
    ("pvt", "ltd"),
    ("private", "ltd"),
    ("pvt", "limited"),
    ("company", "limited"),
    ("co", "ltd"),
    ("co", "llc"),
    ("public", "limited"),
    ("societe", "anonyme"),
    ("societe", "par", "actions", "simplifiee"),
]

SINGLE_TOKEN_SUFFIXES = {
    # US / UK / India
    "llc", "inc", "corp", "corporation", "ltd", "limited", 
    "pvt", "private", "co", "company", "llp", "pllc", "lp", "gmbh",
    # France
    "sarl", "sas", "sasu", "eurl", "sci", "sa", "snc", "sca", "gie", "association",
}

# Regex patterns
RE_ALNUM = re.compile(r"[^a-z0-9]+")
RE_SPACES = re.compile(r"\s+")
RE_US_POSTAL = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
RE_INDIA_POSTAL = re.compile(r"\b(\d{6})\b")
RE_FRANCE_POSTAL = re.compile(r"\b(\d{5})\b")
RE_DIGITS = re.compile(r"\b\d+\b")


def normalize_ascii(text: str) -> str:
    """Transliterates text to clean ASCII using anyascii, strips whitespace, and lowercases."""
    if not text:
        return ""
    translit = anyascii.anyascii(text)
    return translit.strip().lower()


def normalize_alnum(text: str) -> str:
    """Removes non-alphanumeric characters and collapses whitespace."""
    if not text:
        return ""
    cleaned = RE_ALNUM.sub(" ", text.lower())
    return RE_SPACES.sub(" ", cleaned).strip()


def extract_legal_suffix(tokens: List[str]) -> Tuple[List[str], str]:
    """
    Extracts and strips legal suffix from token list.
    Handles suffixes at the end of the name (canonical) or beginning (noisy S3).
    
    Returns:
        (tokens_without_suffix, suffix_string)
    """
    if not tokens:
        return [], ""
        
    n = len(tokens)
    
    # 1. Check multi-token suffixes at the end
    for multi in MULTI_TOKEN_SUFFIXES:
        m_len = len(multi)
        if n >= m_len + 1 and tuple(tokens[-m_len:]) == multi:
            return tokens[:-m_len], " ".join(multi)
            
    # 2. Check multi-token suffixes at the beginning (e.g. S3 noisy records)
    for multi in MULTI_TOKEN_SUFFIXES:
        m_len = len(multi)
        if n >= m_len + 1 and tuple(tokens[:m_len]) == multi:
            return tokens[m_len:], " ".join(multi)
            
    # 3. Check single-token suffix at the end
    if n >= 2 and tokens[-1] in SINGLE_TOKEN_SUFFIXES:
        return tokens[:-1], tokens[-1]
        
    # 4. Check single-token suffix at the beginning
    if n >= 2 and tokens[0] in SINGLE_TOKEN_SUFFIXES:
        return tokens[1:], tokens[0]
        
    return tokens, ""


def extract_address_components(
    addr_ascii: str,
    country: str,
) -> Tuple[str, str, str, str]:
    """
    Extracts structured numeric components from address string.
    
    Returns:
        (addr_alnum, house_number, postal_code, numerics_space_separated)
    """
    if not addr_ascii:
        return "", "", "", ""
        
    addr_alnum = normalize_alnum(addr_ascii)
    digits = RE_DIGITS.findall(addr_alnum)
    
    house_num = digits[0] if digits else ""
    numerics_str = " ".join(digits)
    
    # Extract postal code based on country
    postal = ""
    country_upper = country.upper()
    if country_upper == "US":
        m = RE_US_POSTAL.search(addr_ascii)
        if m:
            postal = m.group(1)
    elif country_upper == "INDIA":
        m = RE_INDIA_POSTAL.search(addr_ascii)
        if m:
            postal = m.group(1)
    elif country_upper == "FRANCE":
        m = RE_FRANCE_POSTAL.search(addr_ascii)
        if m:
            postal = m.group(1)
            
    # Fallback if no regex match: look for 5 or 6 digit number from the end
    if not postal and digits:
        for d in reversed(digits):
            if len(d) in (5, 6):
                postal = d
                break
                
    return addr_alnum, house_num, postal, numerics_str


@dataclass
class NormalizedRecord:
    entity_id: str
    country: str
    source: str
    name_raw: str
    name_ascii: str
    name_alnum: str
    name_no_suffix: str
    legal_suffix: str
    addr_raw: str
    addr_ascii: str
    addr_alnum: str
    house_number: str
    postal_code: str
    addr_numerics: str
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "entity_id": self.entity_id,
            "country": self.country,
            "source": self.source,
            "name_raw": self.name_raw,
            "name_ascii": self.name_ascii,
            "name_alnum": self.name_alnum,
            "name_no_suffix": self.name_no_suffix,
            "legal_suffix": self.legal_suffix,
            "addr_raw": self.addr_raw,
            "addr_ascii": self.addr_ascii,
            "addr_alnum": self.addr_alnum,
            "house_number": self.house_number,
            "postal_code": self.postal_code,
            "addr_numerics": self.addr_numerics,
        }


def normalize_record(
    entity_id: str,
    name_raw: str,
    addr_raw: str,
    country: str,
) -> NormalizedRecord:
    """Executes the full multi-view normalization pipeline for a single record."""
    source = entity_id[:2] if len(entity_id) >= 2 else "UNKNOWN"
    
    # 1. ASCII transliteration
    name_asc = normalize_ascii(name_raw)
    addr_asc = normalize_ascii(addr_raw)
    
    # 2. Alphanumeric cleaning
    name_aln = normalize_alnum(name_asc)
    tokens = name_aln.split() if name_aln else []
    
    # 3. Legal suffix extraction
    tokens_no_suffix, suffix = extract_legal_suffix(tokens)
    name_no_sfx = " ".join(tokens_no_suffix)
    
    # 4. Address decomposition
    addr_aln, house_num, postal, nums_str = extract_address_components(addr_asc, country)
    
    return NormalizedRecord(
        entity_id=entity_id,
        country=country.strip(),
        source=source,
        name_raw=name_raw,
        name_ascii=name_asc,
        name_alnum=name_aln,
        name_no_suffix=name_no_sfx,
        legal_suffix=suffix,
        addr_raw=addr_raw,
        addr_ascii=addr_asc,
        addr_alnum=addr_aln,
        house_number=house_num,
        postal_code=postal,
        addr_numerics=nums_str,
    )
