"""
Unit Tests for Multi-View Text Normalization Pipeline
Amazon ML Challenge 2026
"""

import pytest

from src.entity_resolution.normalization import (
    normalize_ascii,
    normalize_alnum,
    extract_legal_suffix,
    extract_address_components,
    normalize_record,
)


def test_indic_transliteration():
    """Verifies that Indic scripts (Devanagari, Tamil) are properly transliterated."""
    hindi_name = "टाटा कंसल्टेंसी सर्विसेज"
    ascii_out = normalize_ascii(hindi_name)
    assert ascii_out != ""
    assert all(ord(c) < 128 for c in ascii_out)
    assert "tata" in ascii_out


def test_french_diacritics_stripping():
    """Verifies that French accented diacritics are cleanly converted to ASCII."""
    fr_name = "Société Générale de Crédit Électrique"
    ascii_out = normalize_ascii(fr_name)
    assert ascii_out == "societe generale de credit electrique"
    
    fr_addr = "15 Allée des Champs-Élysées, 75008 Paris"
    alnum_out = normalize_alnum(normalize_ascii(fr_addr))
    assert "champs elysees" in alnum_out


def test_legal_suffix_extraction():
    # Canonical English / Indian suffixes
    toks1 = ["reliance", "industries", "limited"]
    clean1, sfx1 = extract_legal_suffix(toks1)
    assert clean1 == ["reliance", "industries"]
    assert sfx1 == "limited"
    
    # Multi-token suffix: pvt ltd
    toks2 = ["infosys", "technologies", "pvt", "ltd"]
    clean2, sfx2 = extract_legal_suffix(toks2)
    assert clean2 == ["infosys", "technologies"]
    assert sfx2 == "pvt ltd"
    
    # US LLC
    toks3 = ["alphabet", "holdings", "llc"]
    clean3, sfx3 = extract_legal_suffix(toks3)
    assert clean3 == ["alphabet", "holdings"]
    assert sfx3 == "llc"
    
    # French SAS
    toks4 = ["kering", "eyewear", "sas"]
    clean4, sfx4 = extract_legal_suffix(toks4)
    assert clean4 == ["kering", "eyewear"]
    assert sfx4 == "sas"
    
    # Reversed suffix prefix from noisy Source 3
    toks5 = ["llc", "hernandez", "colonial", "redwood"]
    clean5, sfx5 = extract_legal_suffix(toks5)
    assert clean5 == ["hernandez", "colonial", "redwood"]
    assert sfx5 == "llc"
    
    # No suffix present
    toks6 = ["redwood", "pharmaceuticals"]
    clean6, sfx6 = extract_legal_suffix(toks6)
    assert clean6 == ["redwood", "pharmaceuticals"]
    assert sfx6 == ""


def test_address_decomposition_us():
    addr = "1600 Amphitheatre Pkwy, Mountain View, CA 94043-1234"
    aln, house_num, postal, nums = extract_address_components(addr, country="US")
    assert house_num == "1600"
    assert postal == "94043"
    assert "1600" in nums
    assert "94043" in nums


def test_address_decomposition_india():
    addr = "Plot No 45, MIDC Industrial Area, Andheri East, Mumbai 400093"
    aln, house_num, postal, nums = extract_address_components(addr, country="India")
    assert house_num == "45"
    assert postal == "400093"
    assert "400093" in nums


def test_address_decomposition_france():
    addr = "10 Rue de la Paix, 75002 Paris"
    aln, house_num, postal, nums = extract_address_components(addr, country="France")
    assert house_num == "10"
    assert postal == "75002"


def test_normalize_record_end_to_end():
    rec = normalize_record(
        entity_id="S1-00042",
        name_raw="L'Oréal Produits de Luxe France SAS",
        addr_raw="14 Rue Royale, 75008 Paris",
        country="France",
    )
    
    assert rec.entity_id == "S1-00042"
    assert rec.country == "France"
    assert rec.source == "S1"
    assert "l oreal" in rec.name_no_suffix or "loreal" in rec.name_no_suffix
    assert rec.legal_suffix == "sas"
    assert rec.house_number == "14"
    assert rec.postal_code == "75008"
    assert rec.to_dict()["legal_suffix"] == "sas"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
