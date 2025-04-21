import functions_framework
from google.cloud import aiplatform
from anthropic import AnthropicVertex
import os
import qdrant_client
import time
import re
import json
import traceback

# Configuration
PROJECT_ID = os.environ.get("QDRANT_CLOUD_URL")
EMBEDDING_LOCATION = os.environ.get("QDRANT_CLOUD_URL")
HAIKU_LOCATION =os.environ.get("QDRANT_CLOUD_URL")
ENDPOINT_ID = os.environ.get("QDRANT_CLOUD_URL")
COLLECTION_NAME = "data_embeddings"

# Initialize clients and services at module level for reuse across invocations
print("Initializing services...")
# Initialize Vertex AI for embeddings
aiplatform.init(project=PROJECT_ID, location=EMBEDDING_LOCATION)

# Create Vertex AI Endpoint for embeddings
endpoint = aiplatform.Endpoint(
    endpoint_name=f"projects/{PROJECT_ID}/locations/{EMBEDDING_LOCATION}/endpoints/{ENDPOINT_ID}"
)

# Initialize Anthropic Vertex client for Claude 3.5 Haiku in us-east5
anthropic_client = AnthropicVertex(region=HAIKU_LOCATION, project_id=PROJECT_ID)

# Qdrant Setup
qdrant = qdrant_client.QdrantClient(
    url=os.environ.get("QDRANT_CLOUD_URL"),
    api_key=os.environ.get("QDRANT_API_KEY"),
    prefer_grpc=False,
    timeout=30
)
print("Services initialized successfully")

class RAGAgent:
    def is_calculation_query(self, query):
        """Identify if the query involves a calculation based on keywords and numbers."""
        calculation_keywords = ["km", "kilometer", "kilometers", "car", "drive", "emission for", "emissions for"]
        return any(keyword in query.lower() for keyword in calculation_keywords) and any(char.isdigit() for char in query)

    def extract_calculation_details(self, query):
        """Extract numerical value (distance in km) from the query using regex."""
        query = query.lower()
        number_match = re.search(r'(\d+\.?\d*)', query)
        if number_match:
            return float(number_match.group(1))
        return None

    def get_embedding(self, text):
        """Generate embedding for the input text using Vertex AI endpoint."""
        instances = [{"inputs": text}]
        response = endpoint.predict(instances=instances)
        embedding = response.predictions[0]
        
        if isinstance(embedding, list) and len(embedding) == 384:
            return embedding
        elif isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list) and len(embedding[0]) == 384:
            return embedding[0]
        else:
            raise ValueError(f"Invalid embedding response: expected a 384-dim list, got {embedding}")

    def search_qdrant(self, query, top_k=3, min_score=0.4):
        """Search Qdrant for relevant context based on query embedding."""
        try:
            query_vector = self.get_embedding(query)
            print(f"Query Vector Length: {len(query_vector)}")
            search_result = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k,
                with_payload=True
            )
            retrieved_texts = [point.payload["text"] for point in search_result.points if point.score > min_score]
            return "\n".join(retrieved_texts) if retrieved_texts else None
        except Exception as e:
            print(f"Qdrant search failed: {str(e)}. Using mock context.")
            return "Mock context: No relevant data found."

    def query_claude(self, prompt, history=None):
        """Query Claude 3.5 Haiku with the provided prompt and conversation history."""
        messages = []
        if history:
            for item in history:
                messages.append({"role": "user", "content": [{"type": "text", "text": item['userInput']}]})
                messages.append({"role": "assistant", "content": [{"type": "text", "text": item['response']}]})
        
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

        for attempt in range(3):  # Retry logic for throttling
            try:
                message = anthropic_client.messages.create(
                    max_tokens=1024,
                    messages=messages,
                    model="claude-3-5-haiku@20241022",
                )
                return message.model_dump_json(indent=2)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < 2:
                    time.sleep(min(1 * (2 ** attempt), 10))  # Exponential backoff
                else:
                    return f"Error: Unable to process query - {str(e)}"

    def process_query(self, query, history=None):
        """Process the user query, handling calculations via LLM or general queries with context."""
        if self.is_calculation_query(query):
            print("Query detected as calculation-related")
            distance = self.extract_calculation_details(query)
            
            if distance:
                calculation_prompt = (
                    f"Query: {query}\n\n"
                    f"You are an ESG Expert tasked with calculating carbon emissions for driving a car. Use the following details:\n"
                    f"- Emission Factor: 0.17001 kg CO₂e per km\n"
                    f"- Formula: Emissions (kg CO₂e) = Distance (km) × Emission Factor (kg CO₂e/km)\n"
                    f"- Distance: {distance} km\n\n"
                    f"Calculate the carbon emissions and provide a clear, concise, and user-friendly explanation of the result. "
                    f"Include the step-by-step calculation and any relevant context based on the conversation history."
                )
                return self.query_claude(calculation_prompt, history)
            else:
                fallback_prompt = (
                    f"Query: {query}\n\n"
                    f"You are an ESG Expert tasked with calculating carbon emissions for driving a car. Use the following details:\n"
                    f"- Emission Factor: 0.17001 kg CO₂e per km\n"
                    f"- Formula: Emissions (kg CO₂e) = Distance (km) × Emission Factor (kg CO₂e/km)\n"
                    f"Unable to parse distance from the query. Based on the query, estimate or explain the carbon emissions "
                    f"for driving a car using the provided emission factor and formula. Provide a detailed, user-friendly response with assumptions."
                )
                return self.query_claude(fallback_prompt, history)
        
        print("Sending query for contextual response with history")
        retrieved_context = self.search_qdrant(query)
        if retrieved_context and retrieved_context.strip():
            final_prompt = (
                f"Query: {query}\n\n"
                f"Use the following information from the database to inform your answer:\n{retrieved_context}\n\n"
                f"Provide a detailed, on-point, and contextual answer based on this information and your general knowledge. "
                f"Use the conversation history to inform your response if relevant."
            )
            print("Using Qdrant context")
        else:
            final_prompt = (
                f"Query: {query}\n\n"
                f"Provide a detailed, on-point, and contextual answer based on your general knowledge, "
                f"as no relevant context was found in the database. "
                f"Use the conversation history to inform your response if relevant."
            )
            print("No relevant context found, using general knowledge")
        return self.query_claude(final_prompt, history)

@functions_framework.http
def handler(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
    Returns:
        The response text, or any set of values that can be turned into a
        Response object.
    """
    print("Function invoked")
    
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    # Set CORS headers for main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    
    try:
        request_json = request.get_json(silent=True)
        print(f"Received request: {json.dumps(request_json) if request_json else 'No JSON data'}")
        
        if not request_json or "query" not in request_json:
            return (json.dumps({"error": "Request must contain a 'query' field"}), 400, headers)
        
        query = request_json.get("query")
        # Standard history
        history = [
            {
                "userInput": "How much CO₂e for a 10 km car trip?",
                "response": "To determine the carbon emissions for a 10 km car trip, let's use the provided details:\n\n- **Emission Factor**: 0.17001 kg CO₂e per km\n- **Formula**: Emissions (kg CO₂e) = Distance (km) × Emission Factor (kg CO₂e/km)\n- **Distance**: 10 km\n\n**Calculation**:\nEmissions = 10 km × 0.17001 kg CO₂e/km = 1.7001 kg CO₂e\n\n**Explanation**: This result represents the carbon dioxide equivalent (CO₂e) emissions produced by driving a car for 10 kilometers. The emission factor of 0.17001 kg CO₂e per km accounts for the average carbon footprint of a car, including fuel combustion and other related factors. This is a straightforward multiplication based on the distance traveled, making it a reliable estimate for a single trip.\n\nSo, a 10 km car trip emits approximately **1.70 kg CO₂e**."
            }
        ]
        # Add any additional history
        if "conversation_history" in request_json:
            history.extend(request_json.get("conversation_history"))
        print(f"Processing query: {query} with {len(history)} history items")
        
        agent = RAGAgent()
        response = agent.process_query(query, history)
        
        return (json.dumps({"result": response}), 200, headers)
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        traceback.print_exc()
        return (json.dumps({"error": f"Error processing query: {str(e)}"}), 500, headers)
