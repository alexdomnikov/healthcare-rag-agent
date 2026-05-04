from docling.document_converter import DocumentConverter

converter = DocumentConverter()
print("Parsing started. This will take 5-15 minutes.")
doc = converter.convert("../data/cms_final_rule.pdf").document

# Export structured representation
print("Parsing completed. Saving to a .json file.")
doc.save_as_json("../data/parsed.json")

print("Done!")