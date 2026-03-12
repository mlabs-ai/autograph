from autograph import autograph
import fire

def main(
    wikidata_json_file: str,
    relationship: str,
    output_file: str
):
    """
    This script takes the WikiData dumps and turns them into smaller DOT files
    for easier processing. The script takes three arguments: a path to the
    original WikiData JSON file dump, a string representing the relationship
    you want to evaluate (e.g., "P171" or "P31"), and a path to the desired
    output location.
    """
    graph = autograph.KnowledgeGraph.from_wikidata(wikidata_json_file, relationship)
    graph.write_to_dot_file(output_file)

if __name__ == "__main__":
    fire.Fire(main)