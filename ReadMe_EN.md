# RWKV Agent Extender

[English](ReadMe_EN.md)	[Chinese](ReadMe.md)

<font color="red" size="3">Note! This is a personal project for practice. If you're looking for a project that can be practically applied, consider projects like [Wenda](https://github.com/wenda-LLM/wenda).</font>

## Description

This is a project aimed at providing RAG and multimodal extension functionalities for RWKV. Here, we aim to achieve the following features:

1. By constructing a local knowledge base, achieve Q&A based on single or multiple documents. It can also retrieve corresponding knowledge from the web, but these document contents will not be saved in the knowledge base. The local knowledge base uses the vector database chroma to store the embedded documents. (Implemented)
2. Additionally, it provides Q&A functionalities for structured data by enabling interactions between the large model and relational databases like MySQL. (Unstable)
3. This project also offers interactive functionalities for image media, allowing image retrieval through natural language descriptions and Q&A based on the text content in images. (Unstable)
4. As a local agent system, we provide a series of tools as extensions to the large model itself. A simple way is provided for users to add new tools. (Not implemented)
5. Voice input (Implemented)

## Installation

<font color="red">Please ensure your LLM is wrapped using tools like FastAPI or can be requested via methods like request. If you are using RWKV, it's recommended to use [RWKV-Runner](https://github.com/josStorer/RWKV-Runner).</font>

Download the project files to your local machine, then use the following command to install dependencies:

```python
pip install -r requirements.txt
```

Then modify `config.json` to replace the parts you need to change with your corresponding values.

Place the rbt6 model (used for building word embeddings) into the rbt6 folder, or you can modify the corresponding part to use other methods.

Place the Clip model into the pretrained_models folder (used for mapping text to images).

Replace the request address of the corresponding LLM in `app.py` with your own address. The default address is [http://127.0.0.1:8000](http://127.0.0.1:8000/).

Run `app.py`. If it runs successfully, it will return the address of a local WebUI.

# Design Concept

This section is mainly for memoranda.

This project centers around the open-source large model RWKV, intending to design a locally deployed agent system to address users' personalized and data privacy issues.

The project uses Gradio as the WebUI development tool and constructs a workflow chain based on Langchain:

1. **Input**: Includes user-provided dialog input and document content input.

   (1) Regarding dialog input:

   Users can input text via keyboard and voice input through a microphone in the WebUI. Voice input is implemented using the whisper model.

   (2) Regarding file input:

   After uploading files in the WebUI, the files will be read.

   If it is a text file, the text content will be embedded by multiple models (including the CLIP model) and stored in different databases.

   If it is structured data, such as csv, xlsx, etc., it will be read by pandas and stored in a relational database (e.g., MySQL).

   If it is an audio document, the whisper model will be called to obtain the text transcription of the audio content, and the text content will be stored as a text file.

   If it is an image document, the CLIP model will be used to embed the image content and store it in the vector database.

2. **Content Processing Chain**:

   (1) Setting Processing Mode:

   The WebUI provides various options to set different Q&A modes for the large model. The modes are as follows, with only the first feature enabled by default:

   1. Enable Q&A based on the local text knowledge base
   2. Enable web search
   3. Refer to structured data content
   4. Refer to audio and image content
   5. Answer based on specific documents
   6. Enable tools

   (2) Q&A Processing Workflow:

   First, check the user's input, including dialog and file input.

   Check if the Q&A based on specific documents is enabled. If so, do not search the knowledge base or web content, and construct a temporary knowledge base. Subsequent content will be based on this temporary knowledge base.

   If not enabled, and Q&A based on the local knowledge base is enabled, the local knowledge base will be used.

   If web search is enabled, both local knowledge base search and web search will be conducted simultaneously, and the most relevant n pieces of content will be saved in a temporary database.

   If referring to audio and image content is enabled, corresponding searches will also be conducted in their respective knowledge bases.

   Based on the above, the n most relevant data blocks will be retrieved from the corresponding databases, and the following operations will be performed:

   If tools are enabled, determine if the user's input requires using tools. If so, the corresponding tools will be enabled, and the results of their execution will be obtained.

   If the user needs to refer to structured data content, the corresponding SQL query will be generated based on the user's input, executed, and the results obtained.

   The above content will be input into the large model, which will then provide an answer.

   (3) Local Knowledge Base Retrieval:

   Text is converted into vectors using word embeddings. Multiple embedding methods are used and stored in different databases. When content is needed, multiple databases are searched simultaneously, and results are filtered, merged, and reordered using LangChain's EnsembleRetriever, MergeRetriever, and LongContextReorder.

   (4) Structured Data Content Processing:

   Structured content is stored in a relational database. When needed, an SQL query is constructed based on the user's query, input into the large model to generate the corresponding SQL query, which is then executed using SQLAlchemy to obtain the result.

   (5) Tools:

   When the LLM wants to use a tool, it first needs to judge the query using IsToolsUsePrompt input into the large model. If it determines that a tool is needed, based on the tools provided in tools.py, a ToolGuidePrompt is constructed, input into the large model, and the corresponding execution code is generated and executed.

## Open-source Projects Used

[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)

[EasyOCR](https://github.com/JaidedAI/EasyOCR)



