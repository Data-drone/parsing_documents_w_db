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
For Prompt develop and testing, the quickest way is to stand up a vllm OpenAI API format server.

You can do that with:
`interactive_examples/Deploy LLM Server`
run that one a single node cluster with GPU compute

VLMs don't operate on a PDF file we need to split the pdfs into separate pages in an image format for the VLM to parse
`interactive_examples/Split Documents`

Then you can use the following notebook in a separate window to send queries to the vllm server
`interactive_examples/Parsing w OpenAI API`

To scale up the workload, ray is the easiest option. Ray is able to optimise batching and IO for best performance. It can also scale up easily on a distributed GPU Cluster on Databricks.
See: `interactive_examples/Parsing w ray`

#### Dev Notes

When using `Parsing w OpenAI API` together with `Deploy LLM Server` on signle node
- Mem leak on LLM Deploy with large (10k+ pages) imagesets
