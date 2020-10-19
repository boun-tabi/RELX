# Datasets


## RELX
This dataset contains 502 parallel sentences in 5 different languages with 18 relations with direction and no_relation in total of 37 categories (18x2+1). English part of this dataset is selected from KBP37 (Zhang and Wang,2015) test set. French, German, Spanish, and Turkish is proposed by our work. We followed file format in KBP37.

In our model, we used the KBP-37 training set as the source dataset. Then, we evaluated on the RELX as the target dataset.

Statistics about the RELX can be seen here:
|            | **Total Sentences** | **Average Characters** | **Average Words** |
|------------|---------------------|------------------------|-------------------|
| **KBP37**  |                     |                        |                   |
| Train      | 15917               | 181\.21                | 30\.28            |
| Validation | 1724                | 181\.77                | 30\.55            |
| Test       | 3405                | 180\.20                | 30\.23            |
| **RELX**   |                     |                        |                   |
| English    | 502                 | 171\.18                | 28\.88            |
| French     | 502                 | 186\.63                | 30\.99            |
| German     | 502                 | 188\.27                | 27\.73            |
| Spanish    | 502                 | 188\.37                | 31\.85            |
| Turkish    | 502                 | 170\.76                | 23\.60            |


## RELX-Distant

This dataset is gathered from Wikipedia and Wikidata.
The process is as follows:

1. The Wikipedia dumps for the corresponding languages are downloaded and converted into raw documents with Wikipedia hyperlinks in entities.
2. The raw documents are split into sentences with spaCy (Honnibal and Montani, 2017), and all hyperlinks are converted to their corresponding Wikidata IDs.
3. Sentences that include entity pairs with Wikidata relations (Vrandečić and Krötzsch, 2014) are collected. We filter and combine some of the relations and propose RELX-Distant whose statistics can be seen in the table below.

| **Language** | **Number of Sentences** |
|--------------|-------------------------|
| English      | 815689                  |
| French       | 652842                  |
| German       | 652062                  |
| Spanish      | 397875                  |
| Turkish      | 57114                   |


### Expanding to New Language
1. Download Wikipedia dump (xml.bz2 file) from https://dumps.wikimedia.org/
2. Convert into txt files with links with [WikiExtractor](https://github.com/attardi/wikiextractor)
```
python3 WikiExtractor.py -o output_folder --processes 8 -l --json dump.xml.bz2
```
3. Use wikimapper to download Wikidata dump and link Wikipedia hyperlinks to Wikidata ID's.
```
wikimapper download {lang}wiki-latest --dir data
wikimapper create {lang}wiki-latest --dumpdir data --target data/index_{lang}wiki-latest.db
```
4. By using Spacy for sentence splitting and wikimapper for extracting relations in sentences, create RELX-Distant dataset for another language.