"""
LLM Interface for GPT-4o-mini Integration
Handles response generation with context management
"""

import asyncio
from typing import List, Dict, Tuple, Optional, Any
import logging
import tiktoken
from datetime import datetime
import os

import openai

logger = logging.getLogger(__name__)

class LLMInterface:
    """GPT-4o-mini interface with advanced prompt engineering"""

    def __init__(self, api_key: str):
        # Set the API key globally
        openai.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key

        # Initialize with sync client only - avoid async issues
        try:
            # Simple, compatible initialization
            self.client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI sync client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None

        self.model = "gpt-4o-mini"
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_context_tokens = 8000  # Conservative limit for gpt-4o-mini
        self.max_response_tokens = 1500

        # Prompt templates
        self.system_prompt = """You are an advanced document analysis assistant with expertise in retrieving and synthesizing information from large documents. Your responses should be:

1.  **Accurate**: Based strictly on the provided context
2.  **Comprehensive**: Cover all relevant aspects found in the sources
3.  **Well-structured**: Use clear organization with headers when appropriate
4.  **Source-aware**: Reference specific sources and pages
5.  **Transparent**: Acknowledge limitations when context is insufficient

Guidelines:
- Always cite sources with page numbers when making claims
- If information is not in the provided context, clearly state this
- Synthesize information from multiple sources when relevant
- Maintain objectivity and avoid speculation beyond the source material
- Use bullet points or numbered lists for complex information when helpful"""

        self.query_templates = {
            "general": """Based on the following document excerpts, please answer this question: {query}

Context from documents:
{context}

Please provide a comprehensive answer with specific citations to page numbers.""",

            "summary": """Please provide a comprehensive summary based on the following document excerpts related to: {query}

Context from documents:
{context}

Include key points, main findings, and important details with proper citations.""",

            # Removed: "comparison" template
            # "comparison": """Compare and analyze the information in these document excerpts regarding: {query}
            # Context from documents:
            # {context}
            # Highlight similarities, differences, and any conflicting information with specific citations.""",

            "analysis": """Analyze the following document excerpts to answer: {query}

Context from documents:
{context}

Provide a detailed analysis including implications, relationships between concepts, and evidence-based conclusions."""
        }

    async def test_connection(self) -> Tuple[bool, str]:
        """Test the OpenAI API connection"""
        try:
            # Use sync client in async context
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
            )
            return True, "Connection successful"
        except Exception as e:
            return False, str(e)

    async def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        response_type: str = "general"
    ) -> str:
        """
        Generate response using retrieved chunks

        Args:
            query: User query
            retrieved_chunks: List of relevant document chunks
            response_type: Type of response (general, summary, analysis)
        """

        try:
            # Build context from chunks
            context = self._build_context(retrieved_chunks)

            # Select appropriate prompt template
            template = self.query_templates.get(response_type, self.query_templates["general"])
            user_prompt = template.format(query=query, context=context)

            # Check token limits and truncate if necessary
            user_prompt = self._manage_context_length(user_prompt)

            # Generate response
            response = await self._call_openai(user_prompt)

            logger.info(f"Generated response for query: {query[:50]}...")
            return response

        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"

    def _build_context(self, chunks: List[Dict]) -> str:
        """Build formatted context from retrieved chunks"""

        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})

            # Extract key metadata
            doc_title = metadata.get('doc_title', 'Unknown Document')
            page_num = metadata.get('page_number', 'Unknown')
            section = metadata.get('section', 'content')
            score = chunk.get('final_score', chunk.get('score', 0))

            # Format chunk
            context_part = f"""
Source {i}: {doc_title} (Page {page_num}, Section: {section}, Relevance: {score:.3f})
Content: {chunk['content'].strip()}
"""
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _manage_context_length(self, prompt: str) -> str:
        """Ensure prompt fits within token limits"""

        tokens = self.tokenizer.encode(prompt)
        available_tokens = self.max_context_tokens - self.max_response_tokens - 100  # Buffer

        if len(tokens) <= available_tokens:
            return prompt

        # Truncate context while preserving query and structure
        lines = prompt.split('\n')

        # Find the context start
        context_start = -1
        for i, line in enumerate(lines):
            if "Context from documents:" in line:
                context_start = i + 1
                break

        if context_start == -1:
            # Fallback: truncate from end
            truncated_tokens = tokens[:available_tokens]
            return self.tokenizer.decode(truncated_tokens)

        # Keep header and query, truncate context
        header_lines = lines[:context_start]
        context_lines = lines[context_start:]

        header_text = '\n'.join(header_lines)
        header_tokens = len(self.tokenizer.encode(header_text))

        available_for_context = available_tokens - header_tokens

        # Truncate context sources
        truncated_context = []
        current_tokens = 0

        for line in context_lines:
            line_tokens = len(self.tokenizer.encode(line))
            if current_tokens + line_tokens > available_for_context:
                break
            truncated_context.append(line)
            current_tokens += line_tokens

        # Add truncation notice
        if len(truncated_context) < len(context_lines):
            truncated_context.append("\n[Note: Some context was truncated due to length limits]")

        return header_text + '\n' + '\n'.join(truncated_context)

    async def _call_openai(self, prompt: str) -> str:
        """Make API call to OpenAI using sync client in async context"""

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

            # Use sync client in async context with executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_response_tokens,
                    temperature=0.1,
                    top_p=0.9
                )
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise

    async def generate_follow_up_questions(
        self,
        query: str,
        response: str,
        retrieved_chunks: List[Dict]
    ) -> List[str]:
        """Generate relevant follow-up questions"""

        try:
            context_summary = self._build_context(retrieved_chunks[:3])  # Use top 3 chunks

            follow_up_prompt = f"""Based on this query and response, suggest 3-5 relevant follow-up questions that would help the user explore the topic further:

Original Query: {query}

Response: {response}

Available Context:
{context_summary}

Generate specific, answerable follow-up questions that:
1.  Explore different aspects of the topic
2.  Could be answered using the available documents
3.  Build upon the information already provided
4.  Are concise and clear

Format as a simple list."""

            follow_ups = await self._call_openai(follow_up_prompt)

            # Parse into list
            questions = []
            for line in follow_ups.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                    # Clean up formatting
                    question = line.lstrip('-•0123456789. ').strip()
                    if question and question.endswith('?'):
                        questions.append(question)

            return questions[:5]

        except Exception as e:
            logger.error(f"Follow-up question generation failed: {str(e)}")
            return []

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        return len(self.tokenizer.encode(text))

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics"""
        return {
            "model": self.model,
            "max_context_tokens": self.max_context_tokens,
            "max_response_tokens": self.max_response_tokens,
            "tokenizer": "cl100k_base",
            "client_type": "OpenAI Sync",
            "features": [
                "single_step_generation",
                "follow_up_questions",
                "context_management"
            ]
        }