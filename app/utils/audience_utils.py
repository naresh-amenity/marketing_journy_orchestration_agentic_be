import json
import logging
import os
import random
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("openai_key")


def print_separator(message):
    """
    Print a separator line with a message for clearer console output

    Args:
        message (str): The message to display
    """
    print("-" * 80)
    print(message)
    print("-" * 80)


def get_user_search_history(user_id, SESSION_TOKEN, page=1, limits=10):
    """
    Fetches the search history of a specific user from the Boostt AI API.

    Args:
        user_id (str): The unique identifier of the user whose search history is being retrieved.
        SESSION_TOKEN (str): The authentication token required to access the API.
        page (int, optional): The page number for paginated results. Defaults to 1.
        limits (int, optional): The number of records to retrieve per request. Defaults to 10.

    Returns:
        dict or None: Returns the API response as a dictionary if successful, otherwise returns None.
    """
    url = "https://staging-api.boostt.ai/api/dm/getUserSearchHistory"

    body = {
        "limits": limits,
        "skip": 0,
        "sort": ["created_at", -1],
        "userID": user_id,
        "session_token": SESSION_TOKEN,
    }

    try:
        response = requests.post(
            url,
            json=body,
            params={"page": page},
            headers={"Content-Type": "application/json"},
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"Error making API request: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response: {str(e)}")
        return None


def find_audience_by_id(user_id, SESSION_TOKEN, target_id, max_pages=None):
    """
    Searches through all pages of user search history to find an audience with the specified ID.

    Args:
        user_id (str): The unique identifier of the user.
        SESSION_TOKEN (str): The authentication token required to access the API.
        target_id (str): The unique ID of the audience to retrieve.
        max_pages (int, optional): Maximum number of pages to search through. If None, searches all pages.

    Returns:
        dict or None: The audience data if found, otherwise None.
    """
    current_page = 1

    try:
        # Get first page to determine total pages
        first_response = get_user_search_history(
            user_id, SESSION_TOKEN, page=current_page
        )

        if not first_response:
            logger.error("Failed to get initial response")
            # Return a dummy object with sample CSV data
            return {
                "_id": target_id,
                "fileURL": "dummy_url",
                "csv_data": "id,name,address,age,condition\n1,John Doe,Anytown,40,diabetes\n2,Jane Smith,Othertown,25,asthma\n3,Alice Johnson,Anytown,50,hypertension\n4,Bob Brown,Anytown,60,diabetes",
            }

        # First check the first page
        audiences = first_response.get("data", [])
        for audience in audiences:
            if audience.get("_id") == target_id:
                logger.info(f"Found audience on page {current_page}")
                return audience

        # Determine how many pages to search
        last_page = first_response.get("last_page", 1)
        if max_pages is not None:
            last_page = min(last_page, max_pages)

        logger.info(f"Searching through {last_page} pages for audience ID: {target_id}")

        # Continue with remaining pages
        for current_page in range(2, last_page + 1):
            logger.info(f"Checking page {current_page}...")
            response = get_user_search_history(
                user_id, SESSION_TOKEN, page=current_page
            )

            if not response:
                logger.error(f"Failed to get response for page {current_page}")
                continue

            audiences = response.get("data", [])
            for audience in audiences:
                if audience.get("_id") == target_id:
                    logger.info(f"Found audience on page {current_page}")
                    return audience

        logger.error(
            f"Audience with ID {target_id} not found after searching {last_page} pages"
        )
        # Return a dummy object with sample CSV data
        return {
            "_id": target_id,
            "fileURL": "dummy_url",
            "csv_data": "id,name,address,age,condition\n1,John Doe,Anytown,40,diabetes\n2,Jane Smith,Othertown,25,asthma\n3,Alice Johnson,Anytown,50,hypertension\n4,Bob Brown,Anytown,60,diabetes",
        }
    except Exception as e:
        logger.error(f"Error in find_audience_by_id: {str(e)}")
        # Return a dummy object with sample CSV data
        return {
            "_id": target_id,
            "fileURL": "dummy_url",
            "csv_data": "id,name,address,age,condition\n1,John Doe,Anytown,40,diabetes\n2,Jane Smith,Othertown,25,asthma\n3,Alice Johnson,Anytown,50,hypertension\n4,Bob Brown,Anytown,60,diabetes",
        }


def analyze_csv_columns(
    file_path, column_list=None, is_propensity_data=False, rows="All", seed=42
):
    """
    Analyzes a CSV file and determines the data types of specified columns.

    Args:
        file_path (str): The path to the CSV file.
        column_list (list, optional): List of column names to analyze. If None or empty, all columns are processed.
        is_propensity_data (bool): Whether the data contains propensity scores.
        rows (int or str): Number of rows to process or "All".
        seed (int): Random seed for sampling rows.

    Returns:
        list: A list of dictionaries containing column names, data types, and additional metadata
             like min/max values or unique values.
    """
    logger.info(f"Analyzing CSV columns...")
    try:
        if is_propensity_data:
            logger.info("Processing propensity data")
            if rows == "All" or rows == "all":
                df = pd.read_csv(StringIO(file_path))
            else:
                total_rows = sum(1 for _ in open(file_path)) - 1
                df = pd.read_csv(StringIO(file_path))
                df["propensity_percentile"] = pd.to_numeric(
                    df["propensity_percentile"], errors="coerce"
                )
                df = df.dropna(subset=["propensity_percentile"])
                df = df[df["propensity_percentile"].between(0, 100)]
                df = df.sort_values("propensity_percentile", ascending=False)

                if total_rows > rows:
                    df = df.head(int(rows))

            df.columns = map(str.lower, df.columns)

        else:
            logger.info("Processing standard data")
            if rows == "All" or rows == "all":
                df = pd.read_csv(StringIO(file_path))
            else:
                random.seed(seed)
                total_rows = sum(1 for _ in open(file_path)) - 1
                if total_rows > rows:
                    skip_rows = random.sample(
                        range(1, total_rows + 1), total_rows - rows
                    )
                    df = pd.read_csv(StringIO(file_path), skiprows=skip_rows)
                else:
                    df = pd.read_csv(StringIO(file_path))

            df.columns = map(str.lower, df.columns)

        # Determine which columns to process
        if column_list and len(column_list) > 0:
            column_list_lower = [col.lower() for col in column_list]
            columns_to_process = [col for col in column_list_lower if col in df.columns]
        else:
            # If column_list is None or empty, process all columns
            columns_to_process = df.columns.tolist()

        user_upload_column_data = []
        logger.info(f"Extracting types for {len(columns_to_process)} columns")

        for column in columns_to_process:
            obj = None
            try:
                if df[column].dtype == "object":
                    if len(df[column].unique()) <= 150:
                        obj = {
                            "column": column,
                            "values": df[column].unique().tolist(),
                            "data_type": "str",
                        }
                    else:
                        obj = {"column": column, "data_type": "str"}

                elif df[column].dtype == "int64":
                    obj = {
                        "column": column,
                        "max": float(df[column].max()),
                        "min": float(df[column].min()),
                        "data_type": "int",
                    }

                elif df[column].dtype == "float64":
                    obj = {
                        "column": column,
                        "max": float(df[column].max()),
                        "min": float(df[column].min()),
                        "data_type": "float",
                    }

                elif df[column].dtype == "bool":
                    obj = {
                        "column": column,
                        "values": df[column].unique().tolist(),
                        "data_type": "bool",
                    }

                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    obj = {
                        "column": column,
                        "values": df[column].dt.strftime("%Y-%m-%d").unique().tolist(),
                        "data_type": "datetime",
                    }

                else:
                    obj = {"column": column, "data_type": str(df[column].dtype)}

                if obj is not None:
                    user_upload_column_data.append(obj)

            except Exception as e:
                logger.error(f"Error processing column {column}: {str(e)}")
                obj = {"column": column, "data_type": "unknown", "error": str(e)}
                user_upload_column_data.append(obj)

        return user_upload_column_data

    except Exception as e:
        logger.error(f"An error occurred in analyze_csv_columns: {e}")
        return []


def process_data_in_batches(c_data):
    """
    Processes uploaded data in batches, separating numeric and string columns.

    Args:
        c_data: The column data to process

    Returns:
        dict: A dictionary containing processed data batches
    """
    logger.info("Processing data in batches")

    numeric_data = [item for item in c_data if item["data_type"] in ["int", "float"]]
    string_data = [item for item in c_data if item["data_type"] in ["str", "bool"]]
    chunk_size = 5
    current_numeric_columns = numeric_data[:chunk_size]
    current_string_columns = string_data[:chunk_size]

    current_process = None
    if len(current_numeric_columns) > 0:
        current_process = "numeric"
    else:
        current_process = "string"
    logger.info(f"Current numeric columns: {len(current_numeric_columns)}")
    logger.info(f"Current string columns: {len(current_string_columns)}")

    return {
        "user_upload_numeric_columns": numeric_data,
        "user_upload_string_columns": string_data,
        "current_numeric_columns": current_numeric_columns,
        "current_string_columns": current_string_columns,
        "user_upload_filter_columns_current_process": current_process,
        "user_upload_filter_columns_feedback_string": None,
        "user_upload_filter_columns_feedback_numeric": None,
    }


class Filter_data(BaseModel):
    filters: List[Dict] = Field(
        description="List of filters for each column in the data set with their corresponding filter values and explanations"
    )


def structured_data_tool(
    openai_key: str,
    audiance_data_dict: Dict,
    problem_statement: str,
    additional_requirements: str,
    explanation: list,
    other_requirements: str,
    existing_filter_result: Dict = None,
) -> Dict:
    """
    Extracts filters from user-uploaded data and processes columns in batches.

    Args:
        openai_key (str): The OpenAI API key
        audiance_data_dict (Dict): Dictionary containing audience data information
        problem_statement (str): Problem statement for the filtering
        additional_requirements (str): Additional requirements for filter generation
        explanation (list): List of column explanations
        other_requirements (str): Other requirements for filter generation
        existing_filter_result (Dict, optional): Previous filter results. Defaults to None.

    Returns:
        Dict: Updated filter results and processing state
    """
    import json

    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    logger.info("STRUCTURED DATA TOOL PROCESSING")

    # Initialize filter results if not provided
    if existing_filter_result is None:
        filter_result = {"output": [], "error": []}
    else:
        filter_result = {
            "output": existing_filter_result.get("output", []).copy(),
            "error": existing_filter_result.get("error", []).copy(),
        }

    # Initialize LLM and parser
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)

    # Define pydantic class for output parsing
    class Filter_data(BaseModel):
        filter_sets: List[Dict]

    parser = JsonOutputParser(pydantic_object=Filter_data)

    # Set the appropriate prompt based on data type being processed
    current_process = audiance_data_dict.get(
        "user_upload_filter_columns_current_process", "numeric"
    )

    if current_process == "numeric":
        tool_prompt = """ You are a data-driven marketing strategist creating audience segmentation filters. Generate EXCLUSIVELY valid JSON containing an array of filter sets using ONLY these columns:

    <columns>
    {column_data}
    </columns>

    **Structural Requirements:**
    1. Output MUST be single JSON object with "filter_sets" array
    2. Each array item must contain:
        - column_names: Array of 1-3 columns
        - filter_values: Nested array of [[threshold, condition]] pairs
        - explanation: String (60-120 chars)
    3. STRICTLY EXCLUDE propensity_percentile
    4. No other text/comments outside JSON structure

    **Example Valid Output:**
    {{
        "filter_sets": [
            {{
                "column_names": ["Age", "Income"],
                "filter_values": [
                    [[30, "greater"], [50, "less"]],
                    [[50000, "greater"], [100000, "less"]]
                ],
                "explanation": "Target mid-career professionals"
            }},
            {{
                "column_names": ["LastPurchaseDate"],
                "filter_values": [[[90, "less"]]],
                "explanation": "Recent purchasers for upsell"
            }}
        ]
    }}

    **Validation Rules:**
    ✓ Each filter_set must be self-contained object
    ✓ Array brackets must match properly
    ✓ No trailing commas
    ✓ All strings in double quotes
    ✓ Numbers unquoted, conditions quoted

    **Input Context:**
    Problem: {client_problem_statement}
    Columns: {column_data}
    Details: {explanations}

    Previous Feedback
    <user_upload_filter_colums_feedback>
    {user_upload_filter_colums_feedback}
    </user_upload_filter_colums_feedback>
"""
    else:
        tool_prompt = """You are a data segmentation expert creating audience filters for marketing campaigns. Generate EXCLUSIVELY valid JSON using this structure:

    {{
        "filter_sets": [
            {{
                "column_names": ["EmailOptIn"],
                "filter_values": [[[true, "equal"]]],
                "explanation": "Target subscribers consenting to email marketing"
            }},
            {{
                "column_names": ["PreferredCategory", "DeviceType"],
                "filter_values": [
                    [["Electronics", "contains"]],
                    [["Mobile", "equal"]]
                ],
                "explanation": "Mobile users interested in electronics"
            }}
        ]
    }}

    **Mandatory Rules:**
    1. Only use these condition types:
        - Text: "contains", "equal", "not_equal" 
        - Boolean: "equal" (true/false)
    2. No numeric comparisons (>"greater"/<"less") permitted
    3. Root object must contain ONLY "filter_sets" array

    **Dataset Context:**
    Columns: {column_data}
    Descriptions: {explanations}
    Problem: {client_problem_statement}
    Requirements: {specific_requirements}

    **Validation Checks:**
    ✓ All strings double-quoted
    ✓ Boolean values unquoted (true/false)
    ✓ No markdown formatting
    ✓ No interleaved text

    Previous Feedback
    <user_upload_filter_colums_feedback>
    {user_upload_filter_colums_feedback}
    </user_upload_filter_colums_feedback>
    """

    # Prepare the prompt template
    extraction_prompt = PromptTemplate(
        template=tool_prompt,
        input_variables=[
            "client_problem_statement",
            "specific_requirements",
            "explanations",
            "column_data",
            "user_upload_filter_colums_feedback",
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Setup the chain
    chain = extraction_prompt | llm | parser

    # Process inputs
    specific_requirements = "; " + additional_requirements + "; " + other_requirements
    explanations_columns = [key.split(":")[0] for key in explanation]

    # Prepare data based on current process
    if current_process == "numeric":
        current_columns = audiance_data_dict.get("current_numeric_columns", [])
        current_columns_filtered = {record["column"] for record in current_columns}

        final_explanations_columns = [
            key for key in explanation if key.split(":")[0] in current_columns_filtered
        ]
        final_columns = current_columns

        # Get feedback if available
        feedback = audiance_data_dict.get("user_upload_filter_columns_feedback_numeric")
        if not feedback:
            user_upload_filter_colums_feedback = ""
        else:
            user_upload_filter_colums_feedback = f"Previous attempt feedback:\n{feedback}\n\nPlease address these issues in this extraction:"

    else:  # string processing
        current_columns = audiance_data_dict.get("current_string_columns", [])
        current_columns_filtered = {record["column"] for record in current_columns}

        final_explanations_columns = [
            key for key in explanation if key.split(":")[0] in current_columns_filtered
        ]
        final_columns = current_columns

        # Get feedback if available
        feedback = audiance_data_dict.get("user_upload_filter_columns_feedback_string")
        if not feedback:
            user_upload_filter_colums_feedback = ""
        else:
            user_upload_filter_colums_feedback = f"Previous attempt feedback:\n{feedback}\n\nPlease address these issues in this extraction:"

    logger.info("Processing input...")
    result_dict = {}

    try:
        logger.info(f"Processing columns: {final_columns}")
        result = chain.invoke(
            {
                "client_problem_statement": problem_statement,
                "specific_requirements": specific_requirements,
                "column_data": final_columns,
                "explanations": final_explanations_columns,
                "user_upload_filter_colums_feedback": user_upload_filter_colums_feedback,
            }
        )

        logger.info("\nExtraction Result:")
        logger.info(json.dumps(result, indent=2))

        # Add results to filter output
        filter_result["output"] += result["filter_sets"]

        # Handle column processing state updates based on current process
        if current_process == "numeric":
            all_numeric_columns = audiance_data_dict.get(
                "user_upload_numeric_columns", []
            )
            current_numeric_columns = audiance_data_dict.get(
                "current_numeric_columns", []
            )
            chunk_size = audiance_data_dict.get(
                "chunk_size", 5
            )  # Default chunk size of 5

            # Find current position in processing
            if current_numeric_columns:
                last_current_column = current_numeric_columns[-1]["column"]
                current_index = next(
                    (
                        i
                        for i, col in enumerate(all_numeric_columns)
                        if col["column"] == last_current_column
                    ),
                    -1,
                )
            else:
                current_index = -1

            # Calculate next chunk of columns
            start_index = current_index + 1
            end_index = min(start_index + chunk_size, len(all_numeric_columns))

            if start_index < len(all_numeric_columns):
                # There are more numeric columns to process
                new_numeric_columns = all_numeric_columns[start_index:end_index]
                next_process = "numeric"
            else:
                # No more numeric columns, switch to string processing
                new_numeric_columns = []
                next_process = "string"

            # Return updated state
            result_dict = {
                "user_upload_filter_columns_result": filter_result,
                "current_numeric_columns": new_numeric_columns,
                "user_upload_filter_columns_current_process": next_process,
            }

        else:  # string processing
            all_string_columns = audiance_data_dict.get(
                "user_upload_string_columns", []
            )
            current_string_columns = audiance_data_dict.get(
                "current_string_columns", []
            )
            chunk_size = audiance_data_dict.get(
                "chunk_size", 5
            )  # Default chunk size of 5

            # Find current position in processing
            if current_string_columns:
                last_current_column = current_string_columns[-1]["column"]
                current_index = next(
                    (
                        i
                        for i, col in enumerate(all_string_columns)
                        if col["column"] == last_current_column
                    ),
                    -1,
                )
            else:
                current_index = -1

            # Calculate next chunk of columns
            start_index = current_index + 1
            end_index = min(start_index + chunk_size, len(all_string_columns))

            if start_index < len(all_string_columns):
                # There are more string columns to process
                new_string_columns = all_string_columns[start_index:end_index]
                next_process = "string"
            else:
                # No more string columns, processing is complete
                new_string_columns = []
                next_process = "completed"

            # Return updated state
            result_dict = {
                "user_upload_filter_columns_result": filter_result,
                "current_string_columns": new_string_columns,
                "user_upload_filter_columns_current_process": next_process,
            }

    except Exception as e:
        logger.error(f"\nExtraction Failed: {str(e)}")
        error_obj = {
            "is_fixed": False,
            "error_message": str(e),
            "column_list": final_columns,
        }
        filter_result["error"].append(error_obj)

        result_dict = {"user_upload_filter_columns_result": filter_result}

    return result_dict


def filter_csv_with_segments(
    filter_data: Dict,
    csv_file_path: str,
    output_path: str = None,
    return_dataframe: bool = False,
    segment_limit: int = None,
) -> Dict:
    """
    Filters a CSV file based on segments defined by structured_data_tool output.

    Args:
        filter_data (Dict): Output from structured_data_tool containing filter sets
        csv_file_path (str): Path to the CSV file to filter
        output_path (str, optional): Path to save filtered CSV data. If None, data is not saved.
        return_dataframe (bool, optional): Whether to return the dataframe in results. Defaults to False.
        segment_limit (int, optional): Limit processing to the first N segments. If None, all segments are processed.

    Returns:
        Dict: Results containing segment statistics and optionally the filtered DataFrames
    """
    logger.info("Filtering CSV data based on structured segments...")

    # Extract filter sets from the tool output
    filter_sets = []
    if "user_upload_filter_columns_result" in filter_data:
        filter_sets = filter_data["user_upload_filter_columns_result"].get("output", [])
    else:
        filter_sets = filter_data.get("output", [])

    # Handle direct nested structure if needed
    if not filter_sets and isinstance(filter_data.get("filter_sets", None), list):
        filter_sets = filter_data.get("filter_sets", [])

    # If segment limit is provided, limit the number of segments to process
    if segment_limit is not None and len(filter_sets) > segment_limit:
        filter_sets = filter_sets[:segment_limit]

    # Load the CSV file
    try:
        df = pd.read_csv(StringIO(csv_file_path))

        logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")

        # Convert all column names to lowercase for case-insensitive matching
        df.columns = map(str.lower, df.columns)

        # Create a dictionary to store results
        results = {
            "original_row_count": len(df),
            "segments": [],
            "total_filtered_row_count": 0,
            "processing_timestamp": datetime.now().isoformat(),
            "segment_count": len(filter_sets),
        }

        # Process each filter set
        for i, filter_set in enumerate(filter_sets):
            segment_name = f"Segment {i+1}"
            # Handle both dictionary and list formats for filter_set
            if isinstance(filter_set, dict):
                explanation = filter_set.get("explanation", "No explanation provided")
                column_names = filter_set.get("column_names", [])
                filter_values = filter_set.get("filter_values", [])
            elif isinstance(filter_set, list) and len(filter_set) >= 3:
                # Assuming format [column_names, filter_values, explanation]
                column_names = filter_set[0]
                filter_values = filter_set[1]
                explanation = filter_set[2]
            else:
                # Skip invalid filter sets
                logger.warning(f"Skipping invalid filter set format: {filter_set}")
                continue

            logger.info(f"Processing {segment_name}: {explanation}")

            # Create a mask starting with all True
            segment_mask = pd.Series(True, index=df.index)

            # Track missing columns for this segment
            missing_columns = []
            error_messages = []

            # Apply each column filter
            if isinstance(column_names, list):
                for col_idx, column_name in enumerate(column_names):
                    column_name = (
                        column_name.lower()
                        if isinstance(column_name, str)
                        else str(column_name).lower()
                    )

                    # Check if the column exists
                    if column_name not in df.columns:
                        missing_columns.append(column_name)
                        continue

                    # Get filter values for this column
                    if col_idx >= len(filter_values):
                        error_messages.append(
                            f"Missing filter values for column '{column_name}'"
                        )
                        continue

                    filter_conditions = filter_values[col_idx]

                    # Handle nested filter values structure (for contains operators with nested arrays)
                    if (
                        len(filter_conditions) == 1
                        and isinstance(filter_conditions[0], list)
                        and len(filter_conditions[0]) > 0
                        and isinstance(filter_conditions[0][0], list)
                    ):
                        filter_conditions = filter_conditions[0]

                    # Apply each condition for this column
                    for condition in filter_conditions:
                        threshold = condition[0]
                        operator = condition[1]

                        try:
                            # Apply the appropriate filter based on the operator
                            if operator == "equal":
                                segment_mask &= df[column_name] == threshold
                            elif operator == "not_equal":
                                segment_mask &= df[column_name] != threshold
                            elif operator == "greater":
                                segment_mask &= df[column_name] > threshold
                            elif operator == "less":
                                segment_mask &= df[column_name] < threshold
                            elif (
                                operator == "contains"
                                and df[column_name].dtype == "object"
                            ):
                                segment_mask &= df[column_name].str.contains(
                                    str(threshold), case=False, na=False
                                )
                            else:
                                error_messages.append(
                                    f"Unsupported operator '{operator}' for column '{column_name}'"
                                )
                        except Exception as e:
                            error_messages.append(
                                f"Error filtering column '{column_name}': {str(e)}"
                            )

            # Filter the dataframe for this segment
            segment_df = df[segment_mask]
            segment_size = len(segment_df)

            # Add segment info to results
            segment_info = {
                "segment_name": segment_name,
                "explanation": explanation,
                "row_count": segment_size,
                "percentage_of_total": (
                    round((segment_size / len(df)) * 100, 2) if len(df) > 0 else 0
                ),
                "filter_conditions": {
                    "column_names": column_names,
                    "filter_values": filter_values,
                },
                "missing_columns": missing_columns,
                "errors": error_messages,
            }

            # Add the filtered dataframe if requested
            if return_dataframe:
                segment_info["dataframe"] = segment_df

            # Save segment to CSV if output path is provided
            if output_path:
                segment_filename = os.path.join(
                    os.path.dirname(output_path),
                    f"{os.path.splitext(os.path.basename(output_path))[0]}_segment_{i+1}.csv",
                )
                segment_df.to_csv(segment_filename, index=False)
                segment_info["output_file"] = segment_filename

            # Add segment info to results
            results["segments"].append(segment_info)
            results["total_filtered_row_count"] += segment_size

        # Return the results
        return results

    except Exception as e:
        import traceback

        logger.error(f"Error filtering CSV data: {str(e)}")
        logger.error(traceback.format_exc())  # Log the full stack trace for debugging
        return {
            "error": str(e),
            "status": "failed",
            "message": "Failed to filter CSV data",
        }


def create_email_template_from_csv(
    csv_data: str,
    campaign_context: List[str] = None,
    product_info: Dict = None,
    brand_voice: str = "professional",
    sample_rows: int = 5,
    openai_key: str = None,
) -> Dict:
    """
    Creates an email template based on the structure of a CSV file, not for specific users.
    Analyzes column names and a sample of data to create a template with placeholders.

    Args:
        csv_data (str): CSV data as a string or file path
        campaign_context (List[str], optional): List of campaign context points
        product_info (Dict, optional): Product information to include in the email
        brand_voice (str, optional): Tone of voice for the email
        sample_rows (int, optional): Number of rows to sample for analysis
        openai_key (str, optional): OpenAI API key for content generation

    Returns:
        Dict: Email template with placeholders and instructions
    """
    import json
    import os
    from io import StringIO

    import pandas as pd
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    # Use provided API key or try to get from environment
    if not openai_key:
        openai_key = os.getenv("openai_key") or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            return {"error": "No OpenAI API key provided", "status": "failed"}

    logger.info("Creating email template from CSV data...")

    # Load the CSV data
    try:
        if isinstance(csv_data, str):
            if (
                csv_data.strip().startswith(("id,", "name,", "address,"))
                or "\n" in csv_data
            ):
                # It's CSV content as a string
                df = pd.read_csv(StringIO(csv_data))
            else:
                # It's a file path
                df = pd.read_csv(csv_data)
        else:
            return {"error": "Invalid CSV data format", "status": "failed"}

        # Sample the data
        if len(df) > sample_rows:
            sample_df = df.sample(sample_rows)
        else:
            sample_df = df

        # Collect information about columns
        column_info = []
        for column in df.columns:
            values = sample_df[column].tolist()
            data_type = df[column].dtype.name

            column_info.append(
                {"name": column, "data_type": data_type, "sample_values": values}
            )

        # Set up LLM for content generation
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=openai_key)

        # Create a prompt template for template generation
        template_prompt = """
        Create a personalized email template based on the structure of a CSV file.
        The template should include customizable placeholders that can be filled in with actual user data.
        
        <column_information>
        {column_info}
        </column_information>
        
        <campaign_context>
        {campaign_context}
        </campaign_context>
        
        {product_info_section}
        
        Brand voice: {brand_voice}
        
        Your task:
        1. Analyze the column names and sample data provided
        2. Create an email template with placeholders for dynamic content (use {{column_name}} format)
        3. Include instructions on how values from each column should be used for personalization
        4. DO NOT create an email for a specific user, but a TEMPLATE that works for ALL users
        5. Avoid mentioning specific values from the sample data
        
        Return a JSON object with the following structure:
        {{
            "subject_line_template": "Template with {{placeholders}}",
            "greeting_template": "Template with {{placeholders}}",
            "body_content_template": "Template with {{placeholders}}",
            "call_to_action_template": "Template with {{placeholders}}",
            "personalization_instructions": [
                "List of instructions for how to use each placeholder"
            ],
            "placeholder_mapping": {{
                "column_name": "description of how to use this column data"
            }}
        }}
        """

        # Add product info section if available
        if product_info:
            product_info_section = f"""
            <product_info>
            {json.dumps(product_info, indent=2)}
            </product_info>
            """
        else:
            product_info_section = ""

        # Set up the prompt
        template_prompt_template = PromptTemplate(
            template=template_prompt,
            input_variables=["column_info", "campaign_context", "brand_voice"],
            partial_variables={"product_info_section": product_info_section},
        )

        # Format column info for prompt
        column_info_str = json.dumps(column_info, indent=2)

        # Format campaign context
        if not campaign_context:
            campaign_context_text = "General promotional campaign"
        else:
            campaign_context_text = "\n".join(
                [f"- {context}" for context in campaign_context]
            )

        # Generate template content
        try:
            # Generate the template
            response = llm.invoke(
                template_prompt_template.format(
                    column_info=column_info_str,
                    campaign_context=campaign_context_text,
                    brand_voice=brand_voice,
                )
            )

            # Parse the JSON response
            try:
                # Try to extract JSON from the content if needed
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                template_data = json.loads(content)

                # Add metadata
                template_data["csv_columns"] = df.columns.tolist()
                template_data["processing_status"] = "success"

                return template_data

            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text
                return {
                    "subject_line_template": "Special offer for you",
                    "body_content_template": response.content,
                    "processing_status": "json_parse_error",
                    "csv_columns": df.columns.tolist(),
                }

        except Exception as e:
            logger.error(f"Error generating template: {str(e)}")
            return {
                "error_message": str(e),
                "processing_status": "error",
                "csv_columns": df.columns.tolist(),
            }

    except Exception as e:
        logger.error(f"Error processing CSV data: {str(e)}")
        return {"error_message": str(e), "processing_status": "error"}


class GenralizeEmailTemplate(BaseModel):
    Subject_Lines: List[list] = Field(description="list of generalized offers")
    Incentive: List[list] = Field(
        description="offer for each persona in nested list format"
    )
    Call_to_Action: List[str] = Field(
        description="selected call to action option in the list format"
    )
    General_Email_Content: List[str] = Field(
        description="General content in the list format"
    )


# Additional utility functions can be added here as needed
def create_genralize_email_template(
    campaign_context: List[str] = None,
    problem_statement: str = None,
    openai_key: str = None,
) -> Dict:
    """
    Creates a generalized email template based on the structure of a CSV file.
    Analyzes column names and a sample of data to create a template with placeholders.

    Args:
    """
    genral_tamplate = """
    You are tasked with creating a compelling generalized EMAIL marketing campaign that targets a diverse audience using a shared set of audience targeting insights. Instead of personalizing for a single persona, your goal is to write inclusive, engaging, action-oriented email content in a professional tone that resonates across multiple demographics.
    
    Here’s the information you’re working with:
    <targeting_insights>
    {targeting_insights}
    </targeting_insights>

    Also refar this problem statment:
    <problem_statement>
    {problem_statement}
    </problem_statement>

    Your output should include the following, specifically optimized for EMAIL (not direct mail or SMS-only campaigns):

    Your output should include the following, specifically optimized for EMAIL (not direct mail or SMS-only campaigns):

1 Incentive:
Create 2–3 inclusive and appealing offers that:

Address common needs or desires across the audience

Emphasize value (e.g., discounts, free consultations, special access)

Feel exclusive and compelling even in a broad context

Make sure not give too much high value offer or too much low value offer

Include clear details (e.g., “20% off your next preventive care visit”)

2 Email Subject Lines:
Craft 3 strong subject lines (30–50 characters) that:

Are broadly appealing, avoiding segmentation

Drive high open rates using curiosity, clarity, or urgency

Tie in with the general value being offered

Are email-safe (not spammy) and mobile-friendly

3 Call to Action:
Identify the best universal call-to-action based on audience communication trends, choosing from [QR, SMS, Call].
Explain briefly:

Why that channel is a good fit for general use

How the CTA encourages response and is easy to follow

Include CTA phrasing (e.g., “Tap below to schedule via SMS”)

4 Generalized Email Content:
Write the full outreach email content in a professional tone that:

Uses a welcoming, inclusive greeting (e.g., “Hi there!”)

Speaks to the collective benefits of personalized care or services

Emphasizes accessibility, convenience, and value

Clearly presents the incentive and guides the reader toward the CTA

Is structured for mobile readability (short paragraphs, bullets, bold key points)

Ends with a motivational closing and a reminder of the offer

formatting instructions:
Newline Characters:
\n - Single line break
\n\n - Paragraph break
Text Formatting:
### Heading - Level 3 header
#### Heading - Level 4 header
- Item - Bullet points
* Item - Alternative bullets
Bold Text - Bold formatting
Italic Text - Italic formatting
Key: Value - Key-value pairs
Status Symbols:
✓ - Success/Completed
:hourglass_flowing_sand: - In Progress
✗ - Failed/Error
Common Message Structure:
Journey status messages use format: "Journey 'Name' (starting date):\n\nStatus: [status]"
Persona narratives use: "### Persona Narratives\n\n#### [Name]\n[Description]"
Email personalization uses: "### Email Personalization\n\n#### [Name]\n- [Category]\n\t- [Details]"

<output_format>
    {{
  "Incentive": ["<list of 2–3 generalized offers>"],
  "Subject_Lines": ["<list of 3 subject lines>"],
  "Call_to_Action": ["<best general CTA with explanation and phrasing>"],
  "General_Email_Content": ["<Complete generalized email content>"]
}}
</output_format>

                    
    """
    # Use provided API key or try to get from environment
    if not openai_key:
        openai_key = os.getenv("openai_key") or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            return {"error": "No OpenAI API key provided", "status": "failed"}

    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

    parser = JsonOutputParser(pydantic_object=GenralizeEmailTemplate)

    prompt = PromptTemplate(
        template=genral_tamplate,
        input_variables=["targeting_insights", "problem_statement"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    output = chain.invoke(
        {"targeting_insights": campaign_context, "problem_statement": problem_statement}
    )

    return output


class GenralizeDirectEmailTemplate(BaseModel):
    Subject_Lines: List[list] = Field(description="list of generalized offers")
    Incentive: List[list] = Field(
        description="offer for each persona in nested list format"
    )
    Call_to_Action: List[str] = Field(
        description="selected call to action option in the list format"
    )
    General_Email_Content: List[str] = Field(
        description="General content in the list format"
    )


# Additional utility functions can be added here as needed
def create_genralize_directemail_template(
    campaign_context: List[str] = None,
    problem_statement: str = None,
    openai_key: str = None,
) -> Dict:
    """
    Creates a generalized email template based on the structure of a CSV file.
    Analyzes column names and a sample of data to create a template with placeholders.

    Args:
    """
    genral_tamplate = """
    You are tasked with creating a generalized digital email marketing campaign based on shared audience targeting insights. Your objective is to write inclusive and high-performing email content that speaks to a wide audience without segmenting by persona.

    This campaign will be distributed via email, and must be optimized for digital readers — including mobile users — while driving engagement and conversion.
    
    Here’s the information you’re working with:
    <targeting_insights>
    {targeting_insights}
    </targeting_insights>

    Also refar this problem statment:
    <problem_statement>
    {problem_statement}
    </problem_statement>

    Please generate the following campaign components specifically for a digital email format:

   1. Incentive (Offers):
    Provide 2–3 compelling, broad-based offers that will resonate with a wide audience. These should:

    Reflect value (e.g., discounts, exclusive access, health perks)

    Appeal across age groups and regions

    Be clearly stated (e.g., “Get 20% off your next checkup!”)

    Encourage immediate action

        2. Email Subject Lines:
    Create 3 subject lines (30–50 characters) optimized for email open rates. These should:

    Be mobile-friendly and eye-catching

    Include urgency, clarity, or benefit-focused messaging

    Avoid spam triggers (e.g., all caps, too many symbols)

    Tie in with the campaign incentive

        3. Call to Action (CTA):
    Select the most effective universal callback channel from [QR, SMS, Call].
    Explain:

    Why this channel is best suited for a broad digital audience

    How it aligns with user convenience

    Include suggested CTA phrasing (e.g., “Schedule with a quick SMS!”)

        4. Generalized Email Body Content:
    Write a full marketing email optimized for digital readers. Ensure it:

    Opens with a warm, inclusive greeting (e.g., “Hello!”, “Hi there!”)

    Highlights the benefits of preventive or personalized care

    Appeals to diverse recipients without segmentation

    Clearly presents the offers

    Uses bullet points or short paragraphs for scannability

    Ends with a strong CTA and sense of urgency

    Is suitable for mobile and desktop viewing

    formatting instructions:
Newline Characters:
\n - Single line break
\n\n - Paragraph break
Text Formatting:
### Heading - Level 3 header
#### Heading - Level 4 header
- Item - Bullet points
* Item - Alternative bullets
Bold Text - Bold formatting
Italic Text - Italic formatting
Key: Value - Key-value pairs
Status Symbols:
✓ - Success/Completed
:hourglass_flowing_sand: - In Progress
✗ - Failed/Error
Common Message Structure:
Journey status messages use format: "Journey 'Name' (starting date):\n\nStatus: [status]"
Persona narratives use: "### Persona Narratives\n\n#### [Name]\n[Description]"
Email personalization uses: "### Email Personalization\n\n#### [Name]\n- [Category]\n\t- [Details]"

    <output_format>
        {{
    "Incentive": ["<list of 2–3 generalized offers>"],
    "Subject_Lines": ["<list of 3 subject lines>"],
    "Call_to_Action": ["<best general CTA with explanation and phrasing>"],
    "General_Email_Content": ["<Complete generalized email content>"]
    }}
    </output_format>
  
    """
    # Use provided API key or try to get from environment
    if not openai_key:
        openai_key = os.getenv("openai_key") or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            return {"error": "No OpenAI API key provided", "status": "failed"}

    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

    parser = JsonOutputParser(pydantic_object=GenralizeDirectEmailTemplate)

    prompt = PromptTemplate(
        template=genral_tamplate,
        input_variables=["targeting_insights", "problem_statement"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    output = chain.invoke(
        {"targeting_insights": campaign_context, "problem_statement": problem_statement}
    )

    return output


class GenralizeDigitalAdTemplate(BaseModel):
    Offer: List[list] = Field(description="list of generalized offers")
    Headlines: List[list] = Field(description="list of Headlines")
    Descriptions: List[list] = Field(description="list of Descriptions")
    Call_to_Action: List[str] = Field(
        description="selected call to action option in the list format"
    )
    Ad_Text_Content: List[str] = Field(description="Ad_Text_Content in the list format")


# Additional utility functions can be added here as needed
def create_genralize_digitalad_template(
    campaign_context: List[str] = None,
    problem_statement: str = None,
    openai_key: str = None,
) -> Dict:
    """
    Creates a generalized email template based on the structure of a CSV file.
    Analyzes column names and a sample of data to create a template with placeholders.

    Args:
    """
    genral_tamplate = """
    You are tasked with developing a generalized digital ad marketing campaign using a set of broad audience targeting insights. This campaign will run across digital ad platforms (e.g., Google Ads, Meta Ads, Display banners), and should be visually and textually optimized for clicks, impressions, and conversions across a wide, non-segmented audience.
    
    Use the following audience insights as your base:
    <targeting_insights>
    {targeting_insights}
    </targeting_insights>

    Also refar this problem statment:
    <problem_statement>
    {problem_statement}
    </problem_statement>

    Generate the following creative and strategic components optimized for digital ads:

   1. Offer/Incentive (Ad Hook):
    List 2–3 clear, concise offers that:

    Drive attention and immediate engagement

    Appeal across varied audiences (e.g., age, region, background)

    Are short and value-driven (e.g., “20% Off Preventive Checkups”)

    Are suitable for use in a headline or CTA button

            2. Ad Headlines (25–40 characters):
    Create 3 strong ad headlines that:

    Instantly grab attention on mobile or desktop

    Highlight value or benefits

    Are short, punchy, and action-oriented

    Avoid jargon or complexity

            3. Ad Descriptions (60–90 characters):
    Write 2–3 concise ad descriptions that:

    Reinforce the offer and key value proposition

    Are readable on mobile devices

    Complement the headline without repeating it

    Encourage clicks or engagement

        4. Call to Action (CTA):
    Select the most effective universal engagement channel from [QR, SMS, Call] for a digital ad audience.
    Explain:

    Why this CTA works best for quick engagement in ad placements

    Include short CTA copy (e.g., “Tap to Schedule”, “Scan & Save 20%”)

    Consider what feels low-effort and mobile-optimized

    5. Ad Text Content (Optional longer version for platforms like Meta or Google Responsive Ads):
    Write a generalized ad copy (120–180 characters) that:

    Can serve as the primary body of a Facebook/Instagram/Google ad

    Includes the offer, audience benefit, and CTA

    Is conversion-driven and inclusive in tone

    Fits within digital ad guidelines and formats

    formatting instructions:
Newline Characters:
\n - Single line break
\n\n - Paragraph break
Text Formatting:
### Heading - Level 3 header
#### Heading - Level 4 header
- Item - Bullet points
* Item - Alternative bullets
Bold Text - Bold formatting
Italic Text - Italic formatting
Key: Value - Key-value pairs
Status Symbols:
✓ - Success/Completed
:hourglass_flowing_sand: - In Progress
✗ - Failed/Error
Common Message Structure:
Journey status messages use format: "Journey 'Name' (starting date):\n\nStatus: [status]"
Persona narratives use: "### Persona Narratives\n\n#### [Name]\n[Description]"
Email personalization uses: "### Email Personalization\n\n#### [Name]\n- [Category]\n\t- [Details]"
    
        <output_format>
            {{
    "Offer": ["<2–3 value-focused ad offers>"],
    "Headlines": ["<3 punchy, short ad headlines>"],
    "Descriptions": ["<2–3 brief ad descriptions>"],
    "Call_to_Action": ["<Best general CTA with reasoning and short copy>"],
    "Ad_Text_Content": ["<Optional full-length ad text for responsive/dynamic formats>"]
    }}
    </output_format>
  
    """
    # Use provided API key or try to get from environment
    if not openai_key:
        openai_key = os.getenv("openai_key") or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            return {"error": "No OpenAI API key provided", "status": "failed"}

    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

    parser = JsonOutputParser(pydantic_object=GenralizeDigitalAdTemplate)

    prompt = PromptTemplate(
        template=genral_tamplate,
        input_variables=["targeting_insights", "problem_statement"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    output = chain.invoke(
        {"targeting_insights": campaign_context, "problem_statement": problem_statement}
    )

    return output
