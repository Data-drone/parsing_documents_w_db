# Parsing Documents with Databricks

Here is a quick guide to parsing documents for use with LLMs on Databricks
The objective is to build a document store for use in RAG.

## The Basics

Start with:

`interactive_examples/Create Document Database`

Once you have your table you can parse it all using basic libs with:

`interactive_examples/Parsing Documents`

We can then use batch processing to quickly create document summaries:

`interactive_examples/Create Document Summaries`

You can now create a Vector Index on the summaries and create a quick bot experience to search docs

## Advanced Parsing

The parsing step is an important part of the process and improving this can be a key step in the process.
We have previously used just a standard PDF Parse lib.

the state of the art is with VLMs that requires standing up a VLM.

`interactive_examples/Deploy LLM Server`

Next is to split the pdfs into separate pages in an image format for the VLM to parse

`interactive_examples/Split Documents`

Then we can use our deployed LLM Server in order to process all the documents

`interactive_examples/Parsing w OpenAI API`



