#!/usr/bin/env python3
import requests
import json
import sys
import re
import os
import time
import importlib.util
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union

# Configuration
OLLAMA_API = "http://localhost:11434/api/generate"
MEMORY_FILE = "q_memory.json"
TOOLS_JSON = "tools_registry.json"
LOG_FILE = f"agent_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

class Logger:
    """Simple logger that outputs to both console and file."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp and level."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_msg = f"{timestamp} - {level.upper()} - {message}"
        
        # Log to console
        print(formatted_msg)
        
        # Log to file
        with open(self.log_file, "a") as f:
            f.write(formatted_msg + "\n")

# Initialize logger
logger = Logger(LOG_FILE)

class ToolResult:
    """Container for tool execution results."""
    
    def __init__(self, tool_name: str, result: Any, success: bool = True, error: str = None):
        self.tool_name = tool_name
        self.result = result
        self.success = success
        self.error = error
        self.timestamp = datetime.now()
    
    def __str__(self) -> str:
        """String representation of the tool result."""
        if self.success:
            return f"Tool '{self.tool_name}' executed successfully"
        else:
            return f"Tool '{self.tool_name}' failed: {self.error}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "result_type": type(self.result).__name__,
            "result_summary": str(self.result)[:100] + "..." if len(str(self.result)) > 100 else str(self.result)
        }

class ToolExecutor:
    """Handles loading and executing tools."""
    
    def __init__(self):
        self.tools = self._load_tools()
        self.results_history = []
    
    def _load_tools(self) -> Dict:
        """Load available tools from JSON registry."""
        try:
            with open(TOOLS_JSON, "r") as f:
                tools = json.load(f)
                logger.log(f"Successfully loaded {len(tools)} tools from {TOOLS_JSON}")
                return tools
        except FileNotFoundError:
            logger.log(f"Error: '{TOOLS_JSON}' not found. Run tool discovery first.", "ERROR")
            return {}
        except json.JSONDecodeError:
            logger.log(f"Error: '{TOOLS_JSON}' is not valid JSON.", "ERROR")
            return {}
    
    def execute_tool(self, tool_name: str, params: List[str]) -> ToolResult:
        """Dynamically loads and executes a tool function."""
        # Skip execution if tool_name is None
        if tool_name is None:
            return ToolResult(tool_name, "No tool name specified", False, "No tool name specified")
            
        tool_info = self.tools.get(tool_name)
        if not tool_info:
            return ToolResult(tool_name, f"Tool '{tool_name}' not found in registry", False, f"Tool not found")

        module_name = tool_info["module"]
        file_path = f"tools/{module_name}.py"

        try:
            # Check if file exists before attempting to load
            if not os.path.exists(file_path):
                return ToolResult(tool_name, f"Module file not found: '{file_path}'", False, "Module file not found")
                
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                return ToolResult(tool_name, f"Failed to create module specification", False, "Module spec creation failed")
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Verify the function exists
            if not hasattr(module, tool_name):
                return ToolResult(tool_name, f"Function not found in module", False, "Function not found")
                
            func = getattr(module, tool_name)
            
            # Process parameters
            processed_params = []
            for param in params:
                # If param references previous tool result, resolve it
                if param.startswith("$") and param[1:] in [r.tool_name for r in self.results_history]:
                    # Get the most recent result for that tool
                    for r in reversed(self.results_history):
                        if r.tool_name == param[1:]:
                            processed_params.append(r.result)
                            break
                else:
                    # Clean up quoted strings
                    if isinstance(param, str) and (param.startswith('"') or param.startswith("'")):
                        param = param.strip('"\'')
                    processed_params.append(param)
            
            logger.log(f"Executing tool: {tool_name} with processed params")
            result = func(*processed_params)
            tool_result = ToolResult(tool_name, result)
            
            # Store the result in history
            self.results_history.append(tool_result)
            return tool_result
            
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.log(error_msg, "ERROR")
            return ToolResult(tool_name, error_msg, False, str(e))
    
    def get_tools_context(self) -> str:
        """Generate a context string listing available tools."""
        if not self.tools:
            return "No tools available."
        
        context = "Available tools:\n"
        for tool_name, tool_info in self.tools.items():
            context += f"- {tool_name}: {tool_info['description']}\n"
            context += f"  Parameters: {', '.join(tool_info['params'])}\n"
        
        return context
    
    def get_recent_results_context(self, limit: int = 5) -> str:
        """Get a summary of recent tool results for context."""
        if not self.results_history:
            return "No previous tool results available."
        
        context = "Recent tool results:\n"
        for result in self.results_history[-limit:]:
            if result.success:
                context += f"- {result.tool_name}: Success (type: {type(result.result).__name__})\n"
            else:
                context += f"- {result.tool_name}: Failed - {result.error}\n"
        
        return context
    
    def clear_history(self):
        """Clear the tool execution history."""
        self.results_history = []

class LLMClient:
    """Client for interacting with the LLM API."""
    
    def __init__(self, model_name: str = "deepseek-r1"):
        self.model_name = model_name
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        try:
            logger.log(f"Generating response using model: {self.model_name}")
            response = requests.post(
                OLLAMA_API, 
                json={"model": self.model_name, "prompt": prompt, "stream": False}
            )
            response.raise_for_status()
            result = response.json().get("response", "")
            logger.log(f"Generated response (length: {len(result)})")
            return result
        except requests.exceptions.RequestException as e:
            error_msg = f"API Error: {str(e)}"
            logger.log(error_msg, "ERROR")
            return f"Failed to generate response: {str(e)}"

class ToolParser:
    """Parses tool calls from text."""
    
    def extract_tool_request(self, text: str, available_tools: Dict) -> Optional[Tuple[str, List[str]]]:
        """Extracts tool name and parameters from CALL_TOOL format."""
        match = re.search(r"CALL_TOOL:\s*(\w+)\((.*?)\)", text)
        if match:
            tool_name = match.group(1)
            
            # Check if tool exists
            if tool_name not in available_tools and tool_name != "tool_name":
                logger.log(f"Warning: Tool '{tool_name}' not found in registry", "WARNING")
                return None
                
            # Handle generic placeholder
            if tool_name == "tool_name":
                logger.log("Warning: Generic placeholder 'tool_name' detected", "WARNING")
                return None
                
            # Parse parameters
            param_str = match.group(2).strip()
            if param_str:
                # Handle quoted strings and commas within them
                params = []
                in_quotes = False
                quote_char = None
                current_param = ""
                
                for char in param_str:
                    if char in ['"', "'"]:
                        if not in_quotes:
                            in_quotes = True
                            quote_char = char
                        elif char == quote_char:
                            in_quotes = False
                        current_param += char
                    elif char == ',' and not in_quotes:
                        params.append(current_param.strip())
                        current_param = ""
                    else:
                        current_param += char
                
                if current_param:
                    params.append(current_param.strip())
            else:
                params = []
                
            logger.log(f"Extracted tool request: {tool_name} with {len(params)} params")
            return tool_name, params
        
        return None

class QLearningAgent:
    """Agent that uses Q-learning to improve solutions over time."""
    
    def __init__(self, model_name: str = "deepseek-r1", learning_rate: float = 0.7, 
                 discount_factor: float = 0.9, max_attempts: int = 3):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.max_attempts = max_attempts
        self.q_table = self._load_memory()
        self.llm_client = LLMClient(model_name)
        self.tool_executor = ToolExecutor()
        self.tool_parser = ToolParser()
        
        logger.log(f"Agent initialized with {len(self.tool_executor.tools)} tools")
    
    def _load_memory(self) -> Dict:
        """Load previous Q-learning memory if it exists."""
        try:
            with open(MEMORY_FILE, "r") as f:
                memory = json.load(f)
                logger.log(f"Loaded Q-learning memory with {len(memory)} entries")
                return memory
        except FileNotFoundError:
            logger.log("No previous Q-learning memory found, starting fresh")
            return {}
    
    def _save_memory(self) -> None:
        """Save Q-learning memory to file."""
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.q_table, f)
            logger.log(f"Saved Q-learning memory with {len(self.q_table)} entries")
    
    def generate_response(self, prompt: str) -> Any:
        """Generate a response, potentially executing tools if requested."""
        if "CALL_TOOL:" in prompt:
            tool_request = self.tool_parser.extract_tool_request(prompt, self.tool_executor.tools)
            if tool_request:
                tool_name, params = tool_request
                return self.tool_executor.execute_tool(tool_name, params).result
        
        return self.llm_client.generate_response(prompt)
    
    def process_multi_tool_query(self, prompt: str) -> str:
        """Process a prompt that might contain multiple sequential tool calls."""
        response = ""
        tool_results = []
        
        # Process the prompt to extract all tool calls
        lines = prompt.split('\n')
        for line in lines:
            if "CALL_TOOL:" in line:
                tool_request = self.tool_parser.extract_tool_request(line, self.tool_executor.tools)
                if tool_request:
                    tool_name, params = tool_request
                    result = self.tool_executor.execute_tool(tool_name, params)
                    tool_results.append({
                        "tool": tool_name,
                        "params": params,
                        "result": result.result if result.success else f"Error: {result.error}",
                        "success": result.success
                    })
                    response += f"Executed {tool_name}: {'Success' if result.success else 'Failed'}\n"
                else:
                    response += line + "\n"
            else:
                response += line + "\n"
        
        # If we have tool results, generate a summary
        if tool_results:
            # Using string concatenation instead of f-string
            summary_prompt = """
            Based on the following tool execution results, generate a comprehensive analysis:
            
            """ + json.dumps(tool_results, default=str, indent=2) + """
            
            Provide a detailed report summarizing what was found, including key insights and recommendations.
            """
            summary = self.llm_client.generate_response(summary_prompt)
            response += "\nAnalysis Summary:\n" + summary
        
        return response
    
    def _generate_perspectives(self, user_prompt: str, num_perspectives: int = 3) -> List[str]:
        """Generate multiple perspectives or approaches to the problem."""
        perspectives = []
        logger.log(f"Generating {num_perspectives} perspectives")
        
        tools_context = self.tool_executor.get_tools_context()
        
        for i in range(num_perspectives):
            logger.log(f"Generating perspective {i+1}/{num_perspectives}")
            # Using escaped curly braces for literal curly braces in example
            perspective_prompt = f"""
            You are an AI assistant tasked with generating a unique perspective or approach to solve the following problem:
            
            {user_prompt}
            
            {tools_context}
            
            Please provide a unique perspective (approach #{i+1}) on how to solve this problem.
            If you need to use a tool, format your response as: CALL_TOOL: tool_name(param1, param2, ...)
            Make sure to only use tool names from the available tools list above.
            
            You can also use the result of a previous tool by using $tool_name as a parameter.
            For example: CALL_TOOL: find_all_elements($parse_html, "form", {{}} )
            
            Make this perspective different from typical approaches. Focus on being creative and thorough.
            """
            
            perspective = self.generate_response(perspective_prompt)
            logger.log(f"Generated perspective {i+1} (length: {len(str(perspective))})")
            perspectives.append(perspective)
            
        return perspectives
    
    def _debate_perspectives(self, user_prompt: str, perspectives: List[str]) -> str:
        """Facilitate a debate between different perspectives to find strengths and weaknesses."""
        logger.log("Debating perspectives")
        
        # Convert perspectives to strings if they aren't already
        string_perspectives = []
        for i, perspective in enumerate(perspectives):
            if not isinstance(perspective, str):
                string_perspectives.append(f"Perspective {i+1}:\n{str(perspective)}")
            else:
                string_perspectives.append(f"Perspective {i+1}:\n{perspective}")
        
        # Join all perspectives
        all_perspectives = "\n\n".join(string_perspectives)
        
        debate_prompt = f"""
        You are an AI moderator tasked with analyzing different perspectives on solving the following problem:
        
        {user_prompt}
        
        Here are the different perspectives proposed:
        
        {all_perspectives}
        
        Please analyze each perspective, identifying its strengths and weaknesses. Consider factors such as:
        1. Effectiveness in addressing the core problem
        2. Creativity and innovation
        3. Practicality and implementation challenges
        4. Use of available tools
        5. Potential limitations or edge cases
        
        After analyzing each perspective, synthesize the best elements of each into a coherent approach.
        """
        
        debate_result = self.generate_response(debate_prompt)
        logger.log(f"Completed debate analysis (length: {len(str(debate_result))})")
        return debate_result
    
    def _synthesize_solution(self, user_prompt: str, debate_result: str) -> str:
        """Synthesize a final solution based on the debate results."""
        logger.log("Synthesizing final solution")
        
        # Get tool context
        tools_context = self.tool_executor.get_tools_context()
        tool_results_context = self.tool_executor.get_recent_results_context()
        
        synthesis_prompt = f"""
        You are an AI problem solver tasked with creating a final solution to the following problem:
        
        {user_prompt}
        
        Based on the analysis of different perspectives:
        
        {debate_result}
        
        {tools_context}
        
        {tool_results_context}
        
        Please synthesize a comprehensive solution that incorporates the strongest elements
        from the analysis while addressing identified weaknesses. Your solution should be:
        
        1. Comprehensive and address all aspects of the problem
        2. Clear and well-structured
        3. Practical and implementable
        4. Well-supported with evidence and reasoning
        
        If you need to use any tools, format your response as: CALL_TOOL: tool_name(param1, param2, ...)
        You can use the result of a previous tool by using $tool_name as a parameter.
        
        IMPORTANT: Only use tool names that are available in the list above.
        """
        
        # First check if this is a multi-tool scenario
        solution = self.process_multi_tool_query(synthesis_prompt)
        
        # If not, generate a standard response
        if not solution.strip():
            solution = self.generate_response(synthesis_prompt)
        
        # If solution is not a string, convert it to a well-formatted analysis
        if not isinstance(solution, str):
            result_type = type(solution).__name__
            logger.log(f"Solution is not a string, it's a {result_type}. Converting to structured analysis.")
            
            # Generate analysis based on the tool result
            analysis_prompt = f"""
            You are an expert security analyst. Based on the results of tool execution,
            please provide a comprehensive analysis. The result is of type {result_type}.
            
            Result summary: {str(solution)[:1000]}...
            
            Generate a detailed security report that includes:
            1. Main findings and vulnerabilities identified
            2. Severity ratings for each issue
            3. Recommendations for fixing each problem
            4. General security best practices
            """
            
            structured_analysis = self.llm_client.generate_response(analysis_prompt)
            return structured_analysis
        
        logger.log(f"Generated solution (length: {len(solution)})")
        return solution
    
    def _evaluate_response(self, response: str, criteria: List[str]) -> float:
        """Evaluate the quality of the response based on extracted criteria."""
        logger.log(f"Evaluating solution against criteria: {criteria}")
        
        # Ensure response is a string
        if not isinstance(response, str):
            response_str = str(response)
        else:
            response_str = response
        
        # Create evaluation prompt with criteria
        criteria_str = ", ".join(criteria)
        evaluation_prompt = f"""
        You are an AI evaluator. Please evaluate the following solution based on how well it addresses these criteria:
        {criteria_str}
        
        Solution to evaluate:
        {response_str[:2000]}  # Limit to first 2000 chars for evaluation
        
        Rate the solution on a scale from 0.0 to 1.0, where:
        - 0.0 = Completely fails to address the criteria
        - 0.5 = Partially addresses the criteria
        - 1.0 = Excellently addresses all criteria
        
        Provide only a numerical score between 0.0 and 1.0 without any explanation.
        """
        
        try:
            score_text = self.llm_client.generate_response(evaluation_prompt).strip()
            # Extract numerical score with regex
            match = re.search(r'([0-9]*[.]?[0-9]+)', score_text)
            if match:
                score = float(match.group(0))
                # Ensure score is within bounds
                final_score = max(0.0, min(score, 1.0))
                logger.log(f"Solution evaluation score: {final_score}")
                return final_score
            else:
                logger.log("Could not extract numerical score, using default", "WARNING")
                return 0.5
        except Exception as e:
            logger.log(f"Evaluation error: {str(e)}", "ERROR")
            return 0.5
    
    def _extract_criteria(self, prompt: str) -> List[str]:
        """Extract key criteria from the user's prompt."""
        logger.log("Extracting evaluation criteria from prompt")
        
        criteria_prompt = f"""
        Analyze the following prompt and identify 3-5 key criteria that should be used to evaluate a good solution.
        Format your response as a comma-separated list of criteria only.
        
        Prompt: {prompt}
        """
        
        criteria_response = self.llm_client.generate_response(criteria_prompt)
        
        # If we get structured criteria, use them
        if "," in criteria_response:
            criteria = [criterion.strip() for criterion in criteria_response.split(",")]
            logger.log(f"Extracted criteria: {criteria}")
            return criteria
        
        # Otherwise extract keywords
        logger.log("Extracting keywords as criteria", "WARNING")
        common_words = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "as", "of", "and", "or", "but"}
        words = [word.lower() for word in re.findall(r'\b\w+\b', prompt)]
        key_words = [word for word in words if word not in common_words and len(word) > 3]
        
        # Ensure we have at least some criteria
        if len(key_words) < 3:
            default_criteria = ["relevance", "completeness", "clarity"] + key_words
            logger.log(f"Using default criteria: {default_criteria}")
            return default_criteria
        
        logger.log(f"Using keyword criteria: {key_words[:5]}")
        return key_words[:5]
    
    def solve_problem(self, user_prompt: str) -> str:
        """Main method to solve a problem using the debating agent approach."""
        logger.log(f"Starting to solve problem (prompt length: {len(user_prompt)})")
        
        # Reset tool history for new problem
        self.tool_executor.clear_history()
        
        # Extract criteria for evaluation
        criteria = self._extract_criteria(user_prompt)
        print(f"🔍 Extracted evaluation criteria: {', '.join(criteria)}")
        
        final_solution = ""
        best_score = 0.0
        attempts = 0

        while attempts < self.max_attempts:
            logger.log(f"Starting attempt {attempts+1}/{self.max_attempts}")
            print(f"🤔 Attempt {attempts+1}/{self.max_attempts}: Generating perspectives...")

            # Generate different perspectives
            perspectives = self._generate_perspectives(user_prompt)
            
            print("🗣️ Debating different approaches...")
            debate_result = self._debate_perspectives(user_prompt, perspectives)

            print("🧠 Synthesizing final solution...")
            current_solution = self._synthesize_solution(user_prompt, debate_result)

            # Evaluate the solution
            quality_score = self._evaluate_response(current_solution, criteria)

            # Update Q-values
            state_key = f"attempt_{attempts}"
            current_q_value = self.q_table.get(state_key, 0)
            max_q_value = max(self.q_table.values(), default=0)
            
            # Q-learning update formula
            new_q_value = current_q_value + \
                self.learning_rate * (quality_score + \
                self.discount_factor * max_q_value - \
                current_q_value)
                
            self.q_table[state_key] = new_q_value
            self._save_memory()

            print(f"📊 Solution quality score: {quality_score:.2f}")
            
            # Keep track of the best solution
            if quality_score > best_score:
                best_score = quality_score
                final_solution = current_solution
                logger.log(f"New best solution found (score: {quality_score:.2f})")

            # If solution is good enough, return it
            if quality_score > 0.7:
                print(f"✅ High-quality solution found (score: {quality_score:.2f})")
                logger.log("High-quality solution found, ending process")
                return final_solution

            print(f"⚠️ Solution quality below threshold (0.7), trying again...")
            logger.log(f"Solution quality {quality_score:.2f} below threshold (0.7)")
            attempts += 1

        print(f"⚠️ Max attempts reached. Returning best solution (score: {best_score:.2f}).")
        logger.log(f"Max attempts reached. Returning best solution (score: {best_score:.2f})")
        return final_solution

def main():
    """Main entry point for the agent."""
    logger.log("Starting agent")
    
    if len(sys.argv) < 2:
        logger.log("Error: No input file provided", "ERROR")
        print("❌ Error: Please provide a filename as an argument.")
        sys.exit(1)

    file_name = sys.argv[1]
    logger.log(f"Input file: {file_name}")

    try:
        with open(file_name, "r", encoding="utf-8") as f:
            user_prompt = f.read().strip()
            logger.log(f"Successfully read input file (length: {len(user_prompt)})")
    except FileNotFoundError:
        logger.log(f"Error: File '{file_name}' not found", "ERROR")
        print(f"❌ Error: File '{file_name}' not found.")
        sys.exit(1)

    # Verify tools setup
    try:
        with open(TOOLS_JSON, "r") as f:
            tools = json.load(f)
        print(f"📋 Loaded {len(tools)} tools from {TOOLS_JSON}")
        logger.log(f"Verified tools registry with {len(tools)} tools")
    except FileNotFoundError:
        logger.log(f"Warning: '{TOOLS_JSON}' not found", "WARNING")
        print(f"⚠️ Warning: '{TOOLS_JSON}' not found. Will proceed without tools.")
    except json.JSONDecodeError:
        logger.log(f"Warning: '{TOOLS_JSON}' is not valid JSON", "WARNING")
        print(f"⚠️ Warning: '{TOOLS_JSON}' is not valid JSON. Will proceed without tools.")

    # Create and run agent
    agent = QLearningAgent()
    solution = agent.solve_problem(user_prompt)

    print("\n=== FINAL SOLUTION ===\n")
    print(solution)
    logger.log("Solution provided to user, ending process")

if __name__ == "__main__":
    main()
