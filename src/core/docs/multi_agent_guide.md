# How to Enable the Multi-Agent RAG System

This document provides instructions for enabling and using the multi-agent RAG system in the AIFAQ project.

## Overview

The multi-agent RAG system enhances the retrieval-augmented generation capabilities of AIFAQ by dividing responsibilities among specialized agents:

1. **Query Understanding Agent**: Analyzes user queries to identify intent, context requirements, and keywords
2. **Retrieval Agent**: Manages document retrieval and relevance scoring
3. **Context Integration Agent**: Combines retrieved documents with conversation history
4. **Response Generation Agent**: Creates the final response using the prepared context
5. **Evaluation Agent**: Assesses response quality and provides feedback for improvement

## Enabling the Multi-Agent Endpoint

To enable the multi-agent endpoint, follow these steps:

1. Make sure all the required dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up NLP dependencies:
   ```bash
   python scripts/setup_nlp_dependencies.py
   ```

3. Import the multi-agent system in the `routes/main.py` file:
   ```python
   from multi_agent_rag import create_multi_agent_rag
   import asyncio
   ```

4. Initialize the multi-agent system after the model initialization:
   ```python
   model, tokenizer, vectordb = initialize_models()
   # Initialize the multi-agent RAG system
   multi_agent_system = create_multi_agent_rag(model, tokenizer, vectordb)
   ```

5. Uncomment the multi-agent endpoint in `routes/main.py`:
   ```python
   @router.post("/query/multi-agent", response_model=DetailedResponseQuery)
   async def answer_query_multi_agent(item: RequestQuery) -> DetailedResponseQuery:
       """
       Generate a response using the multi-agent RAG system.
       
       This endpoint processes the query through the full multi-agent pipeline,
       providing a more contextually relevant and evaluated response.
       """
       try:
           # Generate response using the multi-agent system
           result = await multi_agent_system.generate_response(
               query=item.content,
               session_id=item.id
           )
           
           return DetailedResponseQuery(
               id=item.id,
               message=ResponseMessage(
                   content=result["response"],
                   type=1,
                   id=str(uuid.uuid4()),
               ),
               metadata=result.get("metadata", {})
           )
       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))
   ```

6. Restart the API server.

## Testing the Multi-Agent Endpoint

You can test the endpoint using curl:

```bash
curl --header "Content-Type: application/json" --request POST \
     --data '{"id": "test1", "content": "What is Hyperledger Fabric?"}' \
     http://127.0.0.1:8080/query/multi-agent
```

## Running the Test Scripts

Several test scripts are available to verify the multi-agent system:

### Basic Test
```bash
cd src/core/scripts
python test_multi_agent.py
```

### Low Memory Test
For systems with limited GPU memory:
```bash
python test_multi_agent_low_memory.py
```

### Mock Test
For testing without loading the full model:
```bash
python test_multi_agent_mock.py
```

### Agent Coordination Test
For testing just the coordination logic:
```bash
python test_agent_coordination.py
```

### Comprehensive Test
For running a suite of tests on different query types:
```bash
python test_comprehensive.py --mode mock  # For mock testing
python test_comprehensive.py --mode full  # For full system testing
```

## Troubleshooting

### Common Issues

1. **No documents retrieved**: 
   - Check if the vector database is properly populated using `check_vectordb.py`
   - Reingest documents using `reingest_vectordb.py`
   - Examine logs for the retrieval agent to identify issues

2. **Memory errors**:
   - Use the low-memory version of the test script: `test_multi_agent_low_memory.py`
   - Adjust model quantization parameters in `model.py`
   - Set `device_map="auto"` to allow model to split across devices

3. **NLP errors**:
   - Run `setup_nlp_dependencies.py` to ensure all required NLP models are installed
   - Check if NLTK data is downloaded properly
   - Verify SpaCy models are installed

4. **Poor response quality**:
   - Check evaluation scores in the metadata
   - Examine logs for each agent to identify issues
   - Try adjusting the query to be more specific

### Diagnostic Tools

The following diagnostic tools are available:

- `check_vectordb.py`: Check the health and content of the vector database
- `reingest_vectordb.py`: Reingest documents into the vector database
- `profile_multi_agent.py`: Profile the performance of the multi-agent system

## Performance Optimization

To optimize performance:

1. **Vectordb optimizations**:
   - Adjust the number of documents retrieved in the retrieval agent
   - Consider chunking documents into smaller segments during ingestion

2. **Model optimizations**:
   - Use model quantization (8-bit or 4-bit)
   - Enable flash attention if supported by your GPU

3. **System monitoring**:
   - Use the `profile_multi_agent.py` script to identify bottlenecks
   - Check the logs directory for detailed agent logs

## Frontend Integration

The multi-agent system is integrated with the frontend through:

1. The `USE_MULTI_AGENT` flag in `fetcher.ts`
2. The `MultiAgentInfo` component that displays metadata
3. The `MultiAgentResponse` interface in `multiAgentTypes.ts`

To toggle between the standard and multi-agent mode, update the flag in `fetcher.ts`:

```typescript
// Enable or disable multi-agent mode
export const USE_MULTI_AGENT = true;
```
