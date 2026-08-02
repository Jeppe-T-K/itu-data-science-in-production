---
title: Data Formats
---

# Data Formats

![Data formats](/images/data-and-model-versioning/Big-Data-data-formats.png)

From Testing Big Data Application - Scientific Figure on ResearchGate. Available from: https://www.researchgate.net/figure/Big-Data-data-formats_fig3_339027369 [accessed 1 Aug 2026]

<details><summary style="font-size: 1.5em;">Structured vs Unstructured Data </summary>

- Structured
    - Defined schema
    - Relational, tabular, DB, ...
- Unstructured
    - Variable format
    - Texts, images, ...
- Semi-structured
    - JSON text, ...

<details><summary>And this?</summary>

![Image directory overview](/images/data-and-model-versioning/structured-unstructured-data.png)
</details>
</details>

---

<details>
<summary style="font-size: 1.5em;">Common Formats</summary>

<details>
<summary style="font-size: 1.2em; font-weight: bold;">Tabular/Relational Data</summary>

Structured data organized in tables with rows and columns.

![Tabular Data](/images/data-and-model-versioning/statology-tabular.png)

From https://www.statology.org/tabular-data/

- One row per ID, organised in tables
- Commonly associated files:
    - CSV
    - Parquet
    - ORC
- Easy to query and analyze

</details>
<details>
<summary style="font-size: 1.2em; font-weight: bold;">Images (RAW files, compressed)</summary>

Image data comes in various formats with different storage requirements.

![Nikon RAW File Size Options](/images/data-and-model-versioning/nikon-raw.png)

From https://mcpactions.com/nikon-raw-s-file-size-option/

- RAW files are much larger than compressed formats
- Compression may lose information
- Different formats for different use cases

_How would you store raw and compressed images? Hot and/or cold?_
</details>
<details>
<summary style="font-size: 1.2em; font-weight: bold;">Semi-structured</summary>

```json
{
    "glossary": {
        "title": "example glossary",
		"GlossDiv": {
            "title": "S",
			"GlossList": {
                "GlossEntry": {
                    "ID": "SGML",
					"SortAs": "SGML",
					"GlossTerm": "Standard Generalized Markup Language",
					"Acronym": "SGML",
					"Abbrev": "ISO 8879:1986",
					"GlossDef": {
                        "para": "A meta-markup language, used to create markup languages such as DocBook.",
						"GlossSeeAlso": ["GML", "XML"]
                    },
					"GlossSee": "markup"
                }
            }
        }
    }
}
```

- Some structure, no schema
- Commonly used for configurations

</details>

</details>

