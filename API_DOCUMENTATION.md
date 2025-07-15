# Agentic AI Projects - API Documentation

## Overview

This repository contains comprehensive implementations of agentic AI systems using three major frameworks: **LangGraph**, **LlamaIndex**, and **CrewAI**. This documentation covers all public APIs, functions, components, and their usage patterns.

## Table of Contents

1. [LangGraph Components](#langgraph-components)
2. [LlamaIndex Components](#llamaindex-components) 
3. [CrewAI Components](#crewai-components)
4. [Industrial Projects](#industrial-projects)
5. [Common Dependencies](#common-dependencies)
6. [Installation & Setup](#installation--setup)

---

## LangGraph Components

### 1. Vision Agent System

#### Core Classes

##### `AgentState(TypedDict)`
**Purpose**: Manages conversation state for vision-enabled agents

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

**Fields**:
- `messages`: List of conversation messages with automatic message addition

**Usage Example**:
```python
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

state = AgentState(messages=[])
# Messages are automatically managed through the graph
```

#### Core Functions

##### `extract_text(img_path: str) -> str`
**Purpose**: Extract text from image files using multimodal vision models

**Parameters**:
- `img_path` (str): Local path to image file

**Returns**:
- `str`: Extracted text content from the image

**Example**:
```python
from langchain_openai import ChatOpenAI
import base64

vision_llm = ChatOpenAI(model="gpt-4o")

# Extract text from an image
extracted_text = extract_text("path/to/image.png")
print(extracted_text)
```

**Dependencies**:
- `langchain_openai.ChatOpenAI`
- `langchain_core.messages.HumanMessage`
- `base64` (for image encoding)

##### `divide(a: int, b: int) -> float`
**Purpose**: Mathematical division tool for agent calculations

**Parameters**:
- `a` (int): Numerator
- `b` (int): Denominator

**Returns**:
- `float`: Division result

**Example**:
```python
result = divide(10, 3)  # Returns 3.333...
```

### 2. Email Sorting Agent System

#### Core Classes

##### `EmailState(TypedDict)`
**Purpose**: Manages email processing workflow state

```python
class EmailState(TypedDict):
    email: Dict[str, Any]
    is_spam: Optional[bool]
    spam_reason: Optional[str]
    email_category: Optional[str]
    email_draft: Optional[str]
    messages: List[Dict[str, Any]]
```

**Fields**:
- `email`: Email content dictionary with sender, subject, body
- `is_spam`: Boolean indicating if email is classified as spam
- `spam_reason`: Reason for spam classification
- `email_category`: Email categorization
- `email_draft`: AI-generated response draft
- `messages`: Conversation history

#### Workflow Functions

##### `read_email(state: EmailState) -> Dict`
**Purpose**: Initial email reading and logging

**Parameters**:
- `state`: Current email processing state

**Returns**:
- `Dict`: Empty dict (state update handled by framework)

##### `classify_email(state: EmailState) -> Dict`
**Purpose**: Classify email as spam or legitimate using LLM

**Parameters**:
- `state`: Current email processing state

**Returns**:
- `Dict`: Updated state with spam classification

**Example**:
```python
email_data = {
    "sender": "user@example.com",
    "subject": "Meeting Request",
    "body": "Hi, can we schedule a meeting next week?"
}

state = EmailState(email=email_data, messages=[])
updated_state = classify_email(state)
print(f"Is spam: {updated_state['is_spam']}")
```

##### `handle_spam(state: EmailState) -> Dict`
**Purpose**: Process emails classified as spam

##### `drafting_response(state: EmailState) -> Dict`
**Purpose**: Generate AI draft responses for legitimate emails

##### `notify_mr_wayne(state: EmailState) -> Dict`
**Purpose**: Send notifications for important emails

##### `route_email(state: EmailState) -> str`
**Purpose**: Route emails to appropriate processing paths

**Returns**:
- `str`: Next processing step ("handle_spam" or "draft_response")

#### Graph Construction

```python
from langgraph.graph import StateGraph, START, END

# Build the email processing workflow
workflow = StateGraph(EmailState)
workflow.add_node("read_email", read_email)
workflow.add_node("classify_email", classify_email)
workflow.add_node("handle_spam", handle_spam)
workflow.add_node("draft_response", drafting_response)

# Add conditional routing
workflow.add_conditional_edges(
    "classify_email",
    route_email,
    {
        "handle_spam": "handle_spam",
        "draft_response": "draft_response"
    }
)
```

### 3. Travel Agent System

#### Core Classes

##### `AgentState(TypedDict)`
**Purpose**: Manages travel agent conversation state

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
```

##### `FlightsInput(BaseModel)`
**Purpose**: Input schema for flight search functionality

```python
class FlightsInput(BaseModel):
    departure_city: str = Field(description="City to depart from")
    arrival_city: str = Field(description="City to arrive in")
    departure_date: str = Field(description="Departure date (YYYY-MM-DD)")
    return_date: Optional[str] = Field(description="Return date for round trip")
```

##### `HotelsInput(BaseModel)`
**Purpose**: Input schema for hotel search functionality

```python
class HotelsInput(BaseModel):
    location: str = Field(description="Hotel location/city")
    check_in: str = Field(description="Check-in date (YYYY-MM-DD)")
    check_out: str = Field(description="Check-out date (YYYY-MM-DD)")
    guests: int = Field(description="Number of guests")
```

#### Tool Functions

##### `flights_finder(params: FlightsInput) -> str`
**Purpose**: Search for flight options using SerpAPI

**Parameters**:
- `params`: FlightsInput object with search criteria

**Returns**:
- `str`: Formatted flight search results

##### `hotels_finder(params: HotelsInput) -> str`  
**Purpose**: Search for hotel options using SerpAPI

**Parameters**:
- `params`: HotelsInput object with search criteria

**Returns**:
- `str`: Formatted hotel search results

#### Agent Implementation

##### `Agent` Class
**Purpose**: Main travel agent orchestrator

```python
class Agent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.tools = [flights_finder, hotels_finder]
        
    def call_tools_llm(self, state: AgentState) -> dict:
        """Process user queries and call appropriate tools"""
        
    def invoke_tools(self, state: AgentState) -> dict:
        """Execute tool calls from LLM"""
        
    def email_sender(self, state: AgentState) -> dict:
        """Send travel information via email"""
```

**Usage Example**:
```python
# Initialize travel agent
agent = Agent()

# Process travel query
query = "Find flights from New York to London on 2024-12-15"
response = agent.call_tools_llm({"messages": [{"role": "user", "content": query}]})
```

---

## LlamaIndex Components

### 1. Core RAG System

#### Index Management

##### `VectorStoreIndex`
**Purpose**: Core vector database for document storage and retrieval

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter

# Setup
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Create index
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)
vector_index = VectorStoreIndex(nodes)
```

##### `SimpleDirectoryReader`
**Purpose**: Load documents from directory for processing

```python
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_files=["document.pdf"])
documents = reader.load_data()
print(f"Loaded {len(documents)} document(s)")
```

#### Query Engines

##### `QueryEngine`
**Purpose**: Execute queries against indexed documents

```python
# Basic query engine
query_engine = vector_index.as_query_engine()
response = query_engine.query("What is the main topic?")
print(response)
```

##### `RouterQueryEngine`
**Purpose**: Route queries to appropriate specialized engines

```python
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

# Create specialized tools
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_engine,
    description="Useful for summarization questions"
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for specific fact retrieval"
)

# Route between engines
router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[summary_tool, vector_tool]
)
```

### 2. Agentic Workflows

##### `AgentWorkflow`
**Purpose**: Create sophisticated multi-step agent reasoning

```python
from llama_index.core.agent.workflow import AgentWorkflow

class CustomAgentWorkflow(AgentWorkflow):
    def run(self, query: str) -> str:
        # Custom agent logic
        pass
```

### 3. External Tool Integration

#### Wikipedia Tool

```python
from llama_index.tools.wikipedia import WikipediaToolSpec

wiki_tool = WikipediaToolSpec()
wiki_results = wiki_tool.load_data("Artificial Intelligence")
```

#### ArXiv Tool

```python
from llama_index.tools.arxiv import ArxivToolSpec

arxiv_tool = ArxivToolSpec()
papers = arxiv_tool.search("machine learning", max_results=5)
```

#### Brave Search Tool

```python
from llama_index.tools.brave_search import BraveSearchToolSpec

search_tool = BraveSearchToolSpec(api_key="your_api_key")
results = search_tool.search("latest AI research")
```

---

## CrewAI Components

### 1. Core Agent Framework

#### Agent Class

##### `Agent`
**Purpose**: Individual AI agent with specific role and capabilities

```python
from crewai import Agent

planner = Agent(
    role='Content Planner',
    goal='Plan engaging and factually accurate content on {topic}',
    backstory="You're working on planning a blog article about the topic: {topic}.",
    allow_delegation=False,
    verbose=True
)
```

**Key Parameters**:
- `role`: Agent's function within the crew
- `goal`: Agent's objective and decision-making guide  
- `backstory`: Context for agent's role and goal
- `allow_delegation`: Whether agent can delegate tasks
- `verbose`: Enable detailed logging

#### Task Class

##### `Task`
**Purpose**: Define specific work assignments for agents

```python
from crewai import Task

plan_task = Task(
    description=(
        "1. Prioritize the latest trends, key players, and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering their interests and pain points.\n"
        "3. Develop a detailed content outline including introduction, key points, and call-to-action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document with outline, audience analysis, and SEO keywords.",
    agent=planner
)
```

**Key Parameters**:
- `description`: Detailed task instructions
- `expected_output`: Format and content of deliverable
- `agent`: Assigned agent for the task

#### Crew Class

##### `Crew`
**Purpose**: Orchestrate multiple agents working together

```python
from crewai import Crew

crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan_task, write_task, edit_task],
    verbose=2
)

# Execute the crew
result = crew.kickoff(inputs={'topic': 'AI in healthcare'})
```

### 2. Specialized Implementations

#### Blog Writer System

**Agents**:
- **Content Planner**: Research and outline creation
- **Content Writer**: Article writing and SEO optimization
- **Content Editor**: Review and refinement

**Example Usage**:
```python
# Define agents
planner = Agent(
    role='Content Planner',
    goal='Plan engaging content on {topic}',
    backstory="Expert content strategist",
    allow_delegation=False
)

writer = Agent(
    role='Content Writer', 
    goal='Write insightful article on {topic}',
    backstory="Skilled technical writer",
    allow_delegation=False
)

editor = Agent(
    role='Editor',
    goal='Edit blog post for clarity and engagement',
    backstory="Expert editor with attention to detail",
    allow_delegation=False
)

# Create workflow
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan_task, write_task, edit_task],
    verbose=2
)
```

#### Financial Analysis System

**Agents**:
- **Data Analyst**: Market data analysis
- **Trading Strategy Agent**: Strategy development  
- **Execution Agent**: Trade execution planning
- **Risk Management Agent**: Risk assessment

**Hierarchical Process**:
```python
from crewai import Process

crew = Crew(
    agents=[data_analyst, strategy_agent, execution_agent, risk_agent],
    tasks=[analysis_task, strategy_task, execution_task, risk_task],
    process=Process.hierarchical,
    manager_llm=ChatOpenAI(model="gpt-4")
)
```

### 3. Tool Integration

#### Web Scraping Tools

```python
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Assign tools to agents
researcher = Agent(
    role='Market Researcher',
    tools=[search_tool, scrape_tool],
    # ... other parameters
)
```

#### Custom Tools

```python
from crewai.tools import BaseTool

class SentimentAnalysisTool(BaseTool):
    name: str = "Sentiment Analysis"
    description: str = "Analyze sentiment of given text"
    
    def _run(self, argument: str) -> str:
        # Custom sentiment analysis logic
        return f"Sentiment analysis result for: {argument}"
```

---

## Industrial Projects

### 1. Resume Checker Agent

#### Data Models

##### `WorkExperience(BaseModel)`
```python
class WorkExperience(BaseModel):
    job_title: str
    company: str
    duration: str
    description: str
```

##### `Education(BaseModel)`
```python
class Education(BaseModel):
    degree: str
    institution: str
    year: int
    
    @validator('year')
    def set_year(cls, v):
        return max(v, 1900)
```

##### `Resume(BaseModel)`
```python
class Resume(BaseModel):
    name: str
    email: str
    phone: str
    work_experience: List[WorkExperience]
    education: List[Education]
    skills: List[str]
    
    @classmethod
    def mock(cls):
        return cls(
            name="John Doe",
            email="john@example.com",
            # ... mock data
        )
```

##### `Job(BaseModel)`
```python
class Job(BaseModel):
    title: str
    company: str
    description: str
    requirements: List[str]
    
    @classmethod
    def mock(cls):
        return cls(
            title="Software Engineer",
            company="Tech Corp",
            # ... mock data
        )
```

#### Processing Functions

##### `process_resume() -> Resume`
**Purpose**: Parse and validate resume data

##### `process_job() -> Job`
**Purpose**: Parse and validate job posting data

##### `expert(state: MessagesState) -> dict`
**Purpose**: Provide expert analysis and recommendations

### 2. Job Application Booster

**Tools Used**:
- `SerperDevTool`: Web search capabilities
- `ScrapeWebsiteTool`: Website content extraction
- `FileReadTool`: Local file reading
- `MDXSearchTool`: Markdown document search

**Workflow**:
1. **Profile Analysis**: Analyze candidate profile
2. **Market Research**: Research job market trends
3. **Resume Optimization**: Tailor resume for specific roles
4. **Application Strategy**: Develop application approach

---

## Common Dependencies

### Required Packages

```bash
# LangGraph ecosystem
pip install langgraph langchain_openai langchain_core langchain_huggingface

# LlamaIndex ecosystem  
pip install llama-index llama-index-vector-stores-chroma
pip install llama-index-llms-huggingface-api llama-index-embeddings-huggingface

# CrewAI ecosystem
pip install crewai crewai_tools

# Additional tools
pip install serpapi sendgrid streamlit python-dotenv
```

### Environment Variables

```python
import os

# OpenAI Configuration
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# Search and Communication
os.environ["SERPAPI_API_KEY"] = "your_serpapi_key"
os.environ["SERPER_API_KEY"] = "your_serper_key"
os.environ["SENDGRID_API_KEY"] = "your_sendgrid_key"

# Email Configuration
os.environ["FROM_EMAIL"] = "sender@example.com"
os.environ["TO_EMAIL"] = "recipient@example.com"

# Tracing (Optional)
os.environ["LANGFUSE_PUBLIC_KEY"] = "your_langfuse_public_key"
os.environ["LANGFUSE_SECRET_KEY"] = "your_langfuse_secret_key"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"
```

---

## Installation & Setup

### 1. Clone Repository

```bash
git clone <repository_url>
cd agentic-ai-projects
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install framework-specific packages as needed
pip install langgraph langchain_openai  # For LangGraph projects
pip install llama-index                 # For LlamaIndex projects  
pip install crewai crewai_tools        # For CrewAI projects
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

### 4. Run Examples

```bash
# Run Jupyter notebooks
jupyter notebook

# Or run specific scripts
python scripts/travel_agent.py
streamlit run apps/travel_app.py
```

---

## Usage Patterns

### 1. Basic LangGraph Agent

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    messages: list

def agent_node(state: State):
    # Agent logic here
    return {"messages": state["messages"] + ["processed"]}

# Build graph
graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

# Compile and run
app = graph.compile()
result = app.invoke({"messages": ["Hello"]})
```

### 2. Basic LlamaIndex RAG

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load and index documents
documents = SimpleDirectoryReader("data/").load_data()
index = VectorStoreIndex.from_documents(documents)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic?")
print(response)
```

### 3. Basic CrewAI Workflow

```python
from crewai import Agent, Task, Crew

# Create agent
agent = Agent(
    role="Researcher",
    goal="Research the given topic thoroughly",
    backstory="Expert researcher with analytical skills"
)

# Create task
task = Task(
    description="Research artificial intelligence trends",
    expected_output="Comprehensive research report",
    agent=agent
)

# Create and run crew
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

---

## Best Practices

### 1. Error Handling

```python
try:
    result = crew.kickoff()
except Exception as e:
    print(f"Crew execution failed: {e}")
    # Handle gracefully
```

### 2. State Management

```python
# Always define clear state schemas
class ProcessState(TypedDict):
    input_data: str
    processed_data: Optional[str]
    status: str
    errors: List[str]
```

### 3. Tool Configuration

```python
# Configure tools with proper error handling
@tool
def safe_web_search(query: str) -> str:
    """Safely search the web with fallback handling"""
    try:
        return search_tool.search(query)
    except Exception as e:
        return f"Search failed: {str(e)}"
```

### 4. Memory Management

```python
from langgraph.checkpoint.memory import MemorySaver

# Use memory for stateful conversations
memory = MemorySaver()
app = graph.compile(checkpointer=memory)
```

---

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all required API keys are set in environment variables
2. **Import Errors**: Verify all packages are installed with correct versions
3. **Memory Issues**: Use appropriate chunk sizes for large documents
4. **Rate Limiting**: Implement proper retry logic for API calls

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# CrewAI verbose mode
crew = Crew(agents=[agent], tasks=[task], verbose=2)

# LangGraph visualization
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
```

---

This documentation provides comprehensive coverage of all public APIs, functions, and components in the agentic AI projects repository. Each section includes practical examples and usage instructions to help developers effectively utilize these powerful agent frameworks.