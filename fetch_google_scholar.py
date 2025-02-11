from scholarly import scholarly
import json

DOI = "10.1101/2020.07.07.191551"
JSON_PATH = "citation_count.json"


def get_citation_count(doi):
    try:
        search_query = scholarly.search_pubs(doi)
        paper = next(search_query, None)
        return paper['num_citations'] if paper else None
    except Exception as e:
        print(f"Fetch google scholar did not run as expected: {e}")
        return None


citation_count = get_citation_count(DOI)

if citation_count is not None:
    data = {
        "schemaVersion": 1,
        "label": "Citations",
        "message": str(citation_count),
        "color": "blue"
    }
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Updated citation count: {citation_count}")
else:
    print("Skipping citation count update")
