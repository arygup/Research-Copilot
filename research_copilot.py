# Set up qwen3:1.7b via ollama before running the script below

import os
import json
import requests
import arxiv
import sys
from typing import Dict, Any, List
from pypdf import PdfReader
from dataclasses import dataclass, field
from moya.agents.base_agent import Agent, AgentConfig
from moya.tools.base_tool import BaseTool


def setup_logging():
    os.makedirs(LOGS_DIR, exist_ok=True)
    if os.path.exists("trace.jsonl"):
        os.remove("trace.jsonl")

def log_trace(event_type: str, data: Dict[str, Any]):
    try:
        with open("trace.jsonl", "a") as f:
            log_entry = {"event_type": event_type, "data": data}
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error writing to trace log: {e}")

def log_interaction(filename: str, prompt: str, output: str):
    try:
        with open(os.path.join(LOGS_DIR, filename), "w") as f:
            f.write(f"--- PROMPT ---\n{prompt}\n\n--- OUTPUT ---\n{output}")
    except Exception as e:
        print(f"Error writing interaction log '{filename}': {e}")


# OLLAMA agent class config
@dataclass
class OllamaAgentConfig(AgentConfig):
    base_url: str = "http://localhost:11434"
    llm_config: Dict[str, Any] = field(default_factory=lambda: 
    {
        'model_name': "qwen3:1.7b",                     # Change model here
        'temperature': 0.0,
        'options': {'seed': 42, 'num_ctx': 4096}
    })

# OLLAMA agent class 
class OllamaAgent(Agent):
    def __init__(self, config: OllamaAgentConfig):
        super().__init__(config)
        self.base_url = config.base_url
        self.model_name = config.llm_config.get('model_name')
        self.llm_options = config.llm_config.get('options', {})
        self._check_connection()

    def _check_connection(self):
        log_trace("agent.setup.start", {"agent_name": self.agent_name, "base_url": self.base_url})
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m['name'] for m in models]
            if not any(self.model_name in name for name in model_names):
                raise ValueError(f"Model '{self.model_name}' not found. Please run 'ollama pull {self.model_name}'.")
            log_trace("agent.setup.success", {"agent_name": self.agent_name, "model_found": True})
        except requests.exceptions.RequestException as e:
            log_trace("agent.setup.error", {"agent_name": self.agent_name, "error": str(e)})
            raise ConnectionError(f"Failed to connect to Ollama at {self.base_url}. Is Ollama running? Error: {e}")

    def handle_message(self, message: str, **kwargs) -> str:
        log_trace("agent.handle_message.start", {"agent_name": self.agent_name, "prompt_length": len(message)})
        full_prompt = f"{self.system_prompt}\n\n--- TASK ---\n\n{message}"
        payload = {"model": self.model_name, "prompt": full_prompt, "stream": False, "options": self.llm_options}
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            log_trace("agent.handle_message.success", {"agent_name": self.agent_name, "response_length": len(result)})
            return result
        except requests.exceptions.RequestException as e:
            error_msg = f"[OllamaAgent error: {str(e)}]"
            log_trace("agent.handle_message.error", {"agent_name": self.agent_name, "error": error_msg})
            return error_msg

    def handle_message_stream(self, message: str, **kwargs):
        raise NotImplementedError("Streaming is not required for this co-pilot's workflow.")

# Download research papers from arxiv 
class ArxivSearchTool(BaseTool):
    def __init__(self, download_dir="papers"):
        super().__init__(
            name="ArxivSearchTool",
            description="Searches arXiv and downloads PDF papers.",
            function=self.search_and_download
        )
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)
            
    def search_and_download(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        log_trace("tool.arxiv_search.start", {"query": query, "max_results": max_results})
        try:
            search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
            paper_metadata = []
            for result in search.results():
                try:
                    pdf_path = result.download_pdf(dirpath=self.download_dir)
                    paper_metadata.append({
                        "title": result.title,
                        "authors": ", ".join([author.name for author in result.authors]),
                        "summary": result.summary,
                        "pdf_path": pdf_path,
                    })
                    print(f"Downloaded '{result.title}'")
                except Exception as e:
                    print(f"Could not download paper '{result.title}': {e}")
            log_trace("tool.arxiv_search.success", {"download_count": len(paper_metadata)})
            return paper_metadata
        except Exception as e:
            log_trace("tool.arxiv_search.error", {"error": str(e)})
            print(f"An error occurred while searching arXiv: {e}")
            return []
        

# PDF -> text
class PDFParserTool(BaseTool):
    def __init__(self):
        super().__init__(name="PDFParserTool", description="Parses text from PDF files.", function=self.parse_pdf)

    def parse_pdf(self, file_path: str) -> str:
        log_trace("tool.pdf_parser.start", {"file_path": file_path})
        if not os.path.exists(file_path):
            log_trace("tool.pdf_parser.error", {"error": "File not found"})
            return "[Error: File not found]"
        try:
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            log_trace("tool.pdf_parser.success", {"file_path": file_path, "text_length": len(text)})
            return text
        except Exception as e:
            log_trace("tool.pdf_parser.error", {"file_path": file_path, "error": str(e)})
            return f"[Error parsing PDF: {e}]"

# Vector Store (simple storage)
class SimpleVectorStore:
    def __init__(self):
        self.documents = []
    
    def add_document(self, content: str, metadata: Dict):
        doc_id = len(self.documents)
        self.documents.append({"id": doc_id, "content": content, "metadata": metadata})
        log_trace("vector_store.add", {"doc_id": doc_id, "title": metadata.get("title")})

    def get_all_summaries(self) -> List[Dict]:
        return self.documents

LOGS_DIR = "logs"
setup_logging()
log_trace("main.setup.start", {})
topic = input("Enter a research topic: ")
num_papers = int(input("Enter number of research papers (integer around 5/6): "))

arxiv_tool = ArxivSearchTool()
pdf_tool = PDFParserTool()
vector_store = SimpleVectorStore()

summarizer_agent = OllamaAgent(OllamaAgentConfig(
    agent_name="SummarizerAgent", agent_type="OllamaAgent", description="Summarizes academic papers.",
    system_prompt="You are an expert academic researcher. Given text from a research paper, provide a concise, structured summary with these sections: Problem Statement, Methodology, and Key Results."
))
synthesizer_agent = OllamaAgent(OllamaAgentConfig(
    agent_name="SynthesizerAgent", agent_type="OllamaAgent", description="Synthesizes insights from multiple paper summaries.",
    system_prompt="You are a research analyst. Given several paper summaries, synthesize them. Identify and list: 1. Common Themes & Approaches. 2. Unique or Contrasting Ideas. 3. Research Gaps."
))
writer_agent = OllamaAgent(OllamaAgentConfig(
    agent_name="WriterAgent", agent_type="OllamaAgent", description="Writes a mini survey from synthesized research insights.",
    system_prompt="You are a scientific writer. Using the provided synthesis, write a concise mini survey of around 800 words. Structure it with an introduction, a body discussing the findings, and a conclusion. Use citations like [Paper 1], [Paper 2] .etc. correctly."
))


log_trace("main.setup.complete", {"topic": topic})

print(f"\n Step 1: Searching for papers on '{topic}'")
papers = arxiv_tool.search_and_download(query=topic, max_results=num_papers)
if not papers:
    print("No papers found. Exiting.")
    sys.exit()

print("\n Step 2: Parsing and summarizing papers")
for i, paper_meta in enumerate(papers):
    print(f" Processing '{paper_meta['title']}'")
    paper_text = pdf_tool.parse_pdf(file_path=paper_meta['pdf_path'])
    summary_text = summarizer_agent.handle_message(paper_text[:20000])                                                  # 20000 size limit of paper
    log_interaction(f"summary_{i+1}_{os.path.basename(paper_meta['pdf_path'])}.log", paper_text[:20000], summary_text)
    paper_meta['id'] = i + 1
    vector_store.add_document(summary_text, paper_meta)

all_summaries = vector_store.get_all_summaries()
if not all_summaries:
    print("No summaries were generated. Cannot synthesize. Exiting.")
    sys.exit()
    
synthesis_prompt = "Here are the summaries of several research papers:\n\n"
for summary_doc in all_summaries:
    synthesis_prompt += f"--- Paper {summary_doc['metadata']['id']}: {summary_doc['metadata']['title']} ---\n{summary_doc['content']}\n\n"

synthesized_insights = synthesizer_agent.handle_message(synthesis_prompt)
log_interaction("synthesis.log", synthesis_prompt, synthesized_insights)

print("\n Step 3: Generating the final mini survey")

writer_prompt = f"Topic: {topic}\n\nSynthesized Research Insights:\n{synthesized_insights}\n\nReference list of papers:\n"
for summary_doc in all_summaries:
    writer_prompt += f"[Paper {summary_doc['metadata']['id']}] {summary_doc['metadata']['title']} by {summary_doc['metadata']['authors']}\n"

mini_survey = writer_agent.handle_message(writer_prompt)
log_interaction("final_report_generation.log", writer_prompt, mini_survey)

report_path = "report.md"
with open(report_path, "w") as f:
    f.write(f"# Mini Survey on: {topic}\n\n")
    f.write(mini_survey)

print("END, check report.md")