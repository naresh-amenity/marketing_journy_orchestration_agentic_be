import json
import os
import random
from io import StringIO

# from langchain_core.graph import GraphState
from typing import Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()


# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("openai_key")


def print_separator(message):
    print("-" * 80)
    print(message)


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
        print(f"Error making API request: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {str(e)}")
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

    # Get first page to determine total pages
    first_response = get_user_search_history(user_id, SESSION_TOKEN, page=current_page)

    if not first_response:
        print("Failed to get initial response")
        return None

    # First check the first page
    audiences = first_response.get("data", [])
    for audience in audiences:
        if audience.get("_id") == target_id:
            print(f"Found audience on page {current_page}")
            return audience

    # Determine how many pages to search
    last_page = first_response.get("last_page", 1)
    if max_pages is not None:
        last_page = min(last_page, max_pages)

    print(f"Searching through {last_page} pages for audience ID: {target_id}")

    # Continue with remaining pages
    for current_page in range(2, last_page + 1):
        print(f"Checking page {current_page}...")
        response = get_user_search_history(user_id, SESSION_TOKEN, page=current_page)

        if not response:
            print(f"Failed to get response for page {current_page}")
            continue

        audiences = response.get("data", [])
        for audience in audiences:
            if audience.get("_id") == target_id:
                print(f"Found audience on page {current_page}")
                return audience

    print(f"Audience with ID {target_id} not found after searching {last_page} pages")
    return None


USER_ID_PERSONA = "1246"
SESSION_TOKEN_PERSONA = ""
TARGET_ID = "67c8143f3d37c95cde274fd5"

audience_data = find_audience_by_id(
    user_id=USER_ID_PERSONA, SESSION_TOKEN=SESSION_TOKEN_PERSONA, target_id=TARGET_ID
)

csv_url = audience_data.get("fileURL")
response = requests.get(csv_url)
csv_data = response.text
file_path = csv_data


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
    import random
    from io import StringIO

    import pandas as pd

    print(f"Analyzing CSV columns...")
    if True:
        if is_propensity_data:
            print("Processing propensity data")
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
            print("Processing standard data")
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
        print(f"Extracting types for {len(columns_to_process)} columns")

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
                print(f"Error processing column {column}: {str(e)}")
                obj = {"column": column, "data_type": "unknown", "error": str(e)}
                user_upload_column_data.append(obj)

        return user_upload_column_data

    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     return []


out = analyze_csv_columns(file_path, [], is_propensity_data=False, rows="All", seed=42)
print(out)


def process_data_in_batches(c_data):
    """c_data
    Processes uploaded data in batches, separating numeric and string columns.

    Args:
        openai_api (str): OpenAI API key (currently unused in function).

    Returns:
        function: A function that processes the data in batches and updates state.
    """
    print_separator(f"process_data_in_batches TOOL ")
    # c_data = state["user_upload_column_data"]

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
    print_separator("current_numeric_columns")
    print(current_numeric_columns)

    print_separator("current_string_columns")
    print(current_string_columns)

    return {
        # **state,
        "user_upload_numeric_columns": numeric_data,
        "user_upload_string_columns": string_data,
        "current_numeric_columns": current_numeric_columns,
        "current_string_columns": current_string_columns,
        "user_upload_filter_columns_current_process": current_process,
        "user_upload_filter_columns_feedback_string": None,
        "user_upload_filter_columns_feedback_numeric": None,
    }
    # return data_batches


out = process_data_in_batches(out)
print(out)


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

    print("STRUCTURED DATA TOOL PROCESSING")

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
        tool_prompt = """ YYou are a data-driven marketing strategist creating audience segmentation filters. Generate EXCLUSIVELY valid JSON containing an array of filter sets using ONLY these columns:

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
            [
                "column_names": ["Age", "Income"],
                "filter_values": [
                    [[30, "greater"], [50, "less"]],
                    [[50000, "greater"], [100000, "less"]]
                ],
                "explanation": "Target mid-career professionals"
            ],
            [
                "column_names": ["LastPurchaseDate"],
                "filter_values": [[[90, "less"]]],
                "explanation": "Recent purchasers for upsell"
            ]
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
            [
                "column_names": ["EmailOptIn"],
                "filter_values": [[[true, "equal"]]],
                "explanation": "Target subscribers consenting to email marketing"
            ],
            [
                "column_names": ["PreferredCategory", "DeviceType"],
                "filter_values": [
                    [["Electronics", "contains"]],
                    [["Mobile", "equal"]]
                ],
                "explanation": "Mobile users interested in electronics"
            ]
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

    print("Processing input...")
    result_dict = {}

    try:
        print(f"Processing columns: {final_columns}")
        result = chain.invoke(
            {
                "client_problem_statement": problem_statement,
                "specific_requirements": specific_requirements,
                "column_data": final_columns,
                "explanations": final_explanations_columns,
                "user_upload_filter_colums_feedback": user_upload_filter_colums_feedback,
            }
        )

        print("\nExtraction Result:")
        print(json.dumps(result, indent=2))

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
        print(f"\nExtraction Failed: {str(e)}")
        error_obj = {
            "is_fixed": False,
            "error_message": str(e),
            "column_list": final_columns,
        }
        filter_result["error"].append(error_obj)

        result_dict = {"user_upload_filter_columns_result": filter_result}

    return result_dict


# out = structured_data_tool(openai_key=OPENAI_API_KEY, audiance_data_dict=out, problem_statement="i am doctor and want to do a campaign for my patients",
#                      additional_requirements="", explanation="", other_requirements="")
# print(out)


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
    import json
    import os
    from datetime import datetime

    import pandas as pd

    print("Filtering CSV data based on structured segments...")

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

        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")

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
                print(f"Skipping invalid filter set format: {filter_set}")
                continue

            print(f"Processing {segment_name}: {explanation}")

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

        print(f"Error filtering CSV data: {str(e)}")
        print(traceback.format_exc())  # Print the full stack trace for debugging
        return {
            "error": str(e),
            "status": "failed",
            "message": "Failed to filter CSV data",
        }


def create_personalized_email(
    filter_data: Dict,
    user_data: Dict,
    openai_key: str,
    product_info: Dict = None,
    brand_voice: str = "professional",
) -> Dict:
    """
    Creates personalized email content based on audience filters and user data.

    Args:
        filter_data (Dict): Output from structured_data_tool containing filter sets
        user_data (Dict): User profile data to personalize against
        openai_key (str): OpenAI API key for content generation
        product_info (Dict, optional): Product information to include in the email
        brand_voice (str, optional): Tone of voice for the email. Defaults to "professional"

    Returns:
        Dict: Personalized email content with subject line, body, and metadata
    """
    import json

    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    print("Generating personalized email content...")

    # Extract filter sets that match this user's profile
    matching_segments = []

    if "user_upload_filter_columns_result" in filter_data:
        filter_sets = filter_data["user_upload_filter_columns_result"].get("output", [])
    else:
        filter_sets = filter_data.get("output", [])

    # Determine which segments apply to this user
    for filter_set in filter_sets:
        is_match = True

        # Check if the user data matches all conditions in this filter set
        for i, column_name in enumerate(filter_set.get("column_names", [])):
            if column_name not in user_data:
                is_match = False
                break

            # Get filter values for this column
            filter_values = filter_set.get("filter_values", [])

            # Skip if filter_values doesn't have enough elements
            if i >= len(filter_values):
                is_match = False
                break

            # Get the specific conditions for this column
            filter_conditions = filter_values[i]

            # Handle deeply nested filter values
            # The structure might be: [[[["value", "operator"]]]] or [[["value", "operator"]]]
            if (
                len(filter_conditions) == 1
                and isinstance(filter_conditions[0], list)
                and len(filter_conditions[0]) > 0
                and isinstance(filter_conditions[0][0], list)
            ):
                filter_conditions = filter_conditions[0]

            for condition in filter_conditions:
                # Validate condition structure before processing
                if not isinstance(condition, list) or len(condition) < 2:
                    print(f"Warning: Invalid condition format: {condition}")
                    continue

                threshold = condition[0]
                operator = condition[1]

                # Evaluate filter condition
                if operator == "equal" and user_data[column_name] != threshold:
                    is_match = False
                elif operator == "not_equal" and user_data[column_name] == threshold:
                    is_match = False
                elif operator == "greater" and not (
                    isinstance(user_data[column_name], (int, float))
                    and user_data[column_name] > threshold
                ):
                    is_match = False
                elif operator == "less" and not (
                    isinstance(user_data[column_name], (int, float))
                    and user_data[column_name] < threshold
                ):
                    is_match = False
                elif operator == "contains" and not (
                    isinstance(user_data[column_name], str)
                    and threshold.lower() in user_data[column_name].lower()
                ):
                    is_match = False

        # If all conditions matched, add this segment's explanation to our matching segments
        if is_match:
            matching_segments.append(filter_set.get("explanation", ""))

    # Set up LLM for content generation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=openai_key)

    # Create a prompt template for email generation
    email_prompt = """
    Create a personalized marketing email for a customer who belongs to the following segments:
    
    <segments>
    {segments}
    </segments>
    
    <user_profile>
    {user_profile}
    </user_profile>
    
    {product_info_section}
    
    Brand voice: {brand_voice}
    
    Return a JSON object with the following structure:
    {{
        "subject_line": "Compelling subject line for this user",
        "greeting": "Personalized greeting",
        "body_content": "Main email body with personalization (3-4 paragraphs)",
        "call_to_action": "Clear CTA text",
        "personalization_factors": ["List of personalization elements used"]
    }}
    
    Generate the email to be highly relevant to the user's specific attributes and segments.
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
    email_prompt_template = PromptTemplate(
        template=email_prompt,
        input_variables=["segments", "user_profile", "brand_voice"],
        partial_variables={"product_info_section": product_info_section},
    )

    # Format user profile for prompt
    user_profile_str = json.dumps(user_data, indent=2)

    # Generate email content
    try:
        # Combine matching segments into a string
        segments_text = (
            "\n".join([f"- {segment}" for segment in matching_segments])
            if matching_segments
            else "No specific segments matched"
        )

        # Generate the email content
        response = llm.invoke(
            email_prompt_template.format(
                segments=segments_text,
                user_profile=user_profile_str,
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

            email_data = json.loads(content)

            # Add metadata about segment matching
            email_data["matched_segments"] = matching_segments
            email_data["matching_segment_count"] = len(matching_segments)
            email_data["processing_status"] = "success"

            return email_data

        except json.JSONDecodeError:
            # If JSON parsing fails, return a structured version of the raw text
            return {
                "subject_line": "Special offer for you",
                "body_content": response.content,
                "processing_status": "json_parse_error",
                "matched_segments": matching_segments,
                "matching_segment_count": len(matching_segments),
            }

    except Exception as e:
        print(f"Error generating email: {str(e)}")
        return {
            "subject_line": "Error in email generation",
            "body_content": "An error occurred during personalization.",
            "error_message": str(e),
            "processing_status": "error",
            "matched_segments": matching_segments,
            "matching_segment_count": len(matching_segments),
        }


def create_generalized_personalized_email(
    filter_data: Dict,
    user_data: Dict,
    openai_key: str,
    product_info: Dict = None,
    brand_voice: str = "professional",
) -> Dict:
    """
    Creates generalized personalized email content based on all filter data and user data.
    Instead of treating segments separately, this function creates one unified personalized message.

    Args:
        filter_data (Dict): Output from structured_data_tool containing filter sets
        user_data (Dict): User profile data to personalize against
        openai_key (str): OpenAI API key for content generation
        product_info (Dict, optional): Product information to include in the email
        brand_voice (str, optional): Tone of voice for the email. Defaults to "professional"

    Returns:
        Dict: Personalized email content with subject line, body, and metadata
    """
    import json

    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    print("Generating generalized personalized email content...")

    # Extract all filter criteria for context, not just matching ones
    all_filter_criteria = []

    # Extract filter sets from the tool output
    filter_sets = []
    if "user_upload_filter_columns_result" in filter_data:
        filter_sets = filter_data["user_upload_filter_columns_result"].get("output", [])
    else:
        filter_sets = filter_data.get("output", [])

    # Handle direct nested structure if needed
    if not filter_sets and isinstance(filter_data.get("filter_sets", None), list):
        filter_sets = filter_data.get("filter_sets", [])

    # Extract all filter criteria as context
    for filter_set in filter_sets:
        if (
            isinstance(filter_set, dict)
            and "explanation" in filter_set
            and filter_set["explanation"]
        ):
            all_filter_criteria.append(filter_set["explanation"])
        elif isinstance(filter_set, dict) and "column_names" in filter_set:
            column_names = filter_set.get("column_names", [])
            if column_names:
                criteria_desc = (
                    f"Criteria based on: {', '.join(str(col) for col in column_names)}"
                )
                all_filter_criteria.append(criteria_desc)
        elif isinstance(filter_set, list) and len(filter_set) >= 3:
            # Assuming format [column_names, filter_values, explanation]
            if isinstance(filter_set[2], str) and filter_set[2]:
                all_filter_criteria.append(filter_set[2])
            else:
                column_names = filter_set[0] if isinstance(filter_set[0], list) else []
                if column_names:
                    criteria_desc = f"Criteria based on: {', '.join(str(col) for col in column_names)}"
                    all_filter_criteria.append(criteria_desc)

    # Set up LLM for content generation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=openai_key)

    # Create a prompt template for email generation
    email_prompt = """
    Create a personalized marketing email for a customer. Instead of using specific segments, 
    create a cohesive message that feels personally tailored to this individual customer.
    
    <campaign_context>
    This is part of a campaign where we're considering customers with the following characteristics:
    {filter_criteria}
    </campaign_context>
    
    <user_profile>
    {user_profile}
    </user_profile>
    
    {product_info_section}
    
    Brand voice: {brand_voice}
    
    Important guidelines:
    1. DO NOT mention "segments" or "filters" in the email
    2. Make the content feel naturally personalized based on the user's profile
    3. Create a cohesive message that addresses the user holistically
    4. Include specific details from their profile that make the email feel unique to them
    
    Return a JSON object with the following structure:
    {{
        "subject_line": "Compelling subject line for this user",
        "greeting": "Personalized greeting",
        "body_content": "Main email body with personalization (3-4 paragraphs)",
        "call_to_action": "Clear CTA text",
        "personalization_factors": ["List of personalization elements used"]
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
    email_prompt_template = PromptTemplate(
        template=email_prompt,
        input_variables=["filter_criteria", "user_profile", "brand_voice"],
        partial_variables={"product_info_section": product_info_section},
    )

    # Format user profile for prompt
    user_profile_str = json.dumps(user_data, indent=2)

    # Generate email content
    try:
        # Combine filter criteria into a string
        filter_criteria_text = (
            "\n".join([f"- {criteria}" for criteria in all_filter_criteria])
            if all_filter_criteria
            else "No specific criteria defined"
        )

        # Generate the email content
        response = llm.invoke(
            email_prompt_template.format(
                filter_criteria=filter_criteria_text,
                user_profile=user_profile_str,
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

            email_data = json.loads(content)

            # Add metadata
            email_data["filter_criteria_used"] = all_filter_criteria
            email_data["processing_status"] = "success"

            return email_data

        except json.JSONDecodeError:
            # If JSON parsing fails, return a structured version of the raw text
            return {
                "subject_line": "Special offer for you",
                "body_content": response.content,
                "processing_status": "json_parse_error",
                "filter_criteria_used": all_filter_criteria,
            }

    except Exception as e:
        print(f"Error generating email: {str(e)}")
        return {
            "subject_line": "Error in email generation",
            "body_content": "An error occurred during personalization.",
            "error_message": str(e),
            "processing_status": "error",
            "filter_criteria_used": all_filter_criteria,
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

    print("Creating email template from CSV data...")

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
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=openai_key)

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
            print(f"Error generating template: {str(e)}")
            return {
                "error_message": str(e),
                "processing_status": "error",
                "csv_columns": df.columns.tolist(),
            }

    except Exception as e:
        print(f"Error processing CSV data: {str(e)}")
        return {"error_message": str(e), "processing_status": "error"}


def generate_email_content(
    user_data: Dict,
    campaign_context: List[str] = None,
    product_info: Dict = None,
    brand_voice: str = "professional",
    openai_key: str = None,
) -> Dict:
    """
    Standalone function to generate personalized email content for a user.

    Args:
        user_data (Dict): User profile data to personalize against
        campaign_context (List[str], optional): List of campaign context points/criteria
        product_info (Dict, optional): Product information to include in the email
        brand_voice (str, optional): Tone of voice for the email. Defaults to "professional"
        openai_key (str, optional): OpenAI API key for content generation

    Returns:
        Dict: Generated email content with subject line, body, and metadata
    """
    import json
    import os

    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    # Use provided API key or try to get from environment
    if not openai_key:
        openai_key = os.getenv("openai_key") or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            return {"error": "No OpenAI API key provided", "status": "failed"}

    print("Generating standalone email content...")

    # Set up LLM for content generation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=openai_key)

    # Create a prompt template for email generation
    email_prompt = """
    Create a personalized marketing email for a customer based on their profile and campaign context.
    
    <campaign_context>
    {campaign_context}
    </campaign_context>
    
    <user_profile>
    {user_profile}
    </user_profile>
    
    {product_info_section}
    
    Brand voice: {brand_voice}
    
    Important guidelines:
    1. DO NOT mention "segments" or "filters" in the email
    2. Make the content feel naturally personalized based on the user's profile
    3. Create a cohesive message that addresses the user holistically
    4. Include specific details from their profile that make the email feel unique to them
    5. Make the email concise but impactful
    
    Return a JSON object with the following structure:
    {{
        "subject_line": "Compelling subject line for this user",
        "greeting": "Personalized greeting",
        "body_content": "Main email body with personalization (3-4 paragraphs)",
        "call_to_action": "Clear CTA text",
        "personalization_factors": ["List of personalization elements used"]
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
    email_prompt_template = PromptTemplate(
        template=email_prompt,
        input_variables=["campaign_context", "user_profile", "brand_voice"],
        partial_variables={"product_info_section": product_info_section},
    )

    # Format user profile for prompt
    user_profile_str = json.dumps(user_data, indent=2)

    # Format campaign context
    if not campaign_context:
        campaign_context_text = "General promotional campaign"
    else:
        campaign_context_text = "\n".join(
            [f"- {context}" for context in campaign_context]
        )

    # Generate email content
    try:
        # Generate the email content
        response = llm.invoke(
            email_prompt_template.format(
                campaign_context=campaign_context_text,
                user_profile=user_profile_str,
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

            email_data = json.loads(content)

            # Add metadata
            email_data["processing_status"] = "success"

            return email_data

        except json.JSONDecodeError:
            # If JSON parsing fails, return a structured version of the raw text
            return {
                "subject_line": "Special offer for you",
                "body_content": response.content,
                "processing_status": "json_parse_error",
            }

    except Exception as e:
        print(f"Error generating email: {str(e)}")
        return {
            "subject_line": "Error in email generation",
            "body_content": "An error occurred during personalization.",
            "error_message": str(e),
            "processing_status": "error",
        }


# Example of how to use the standalone email generation function
def demo_standalone_email_generation():
    """
    Demonstrates how to use the generate_email_content function independently.
    """
    # Sample user data
    user_data = {
        "name": "John Doe",
        "address": "123 Main St, Anytown, CA",
        "age": 45,
        "condition": "diabetes",
        "last_visit": "2023-10-15",
        "email": "john.doe@example.com",
    }

    # Sample campaign context
    campaign_context = [
        "Health and wellness campaign targeting adults with chronic conditions",
        "Focus on preventative care and regular check-ups",
        "Promoting new telehealth services and extended office hours",
    ]

    # Sample product information
    product_info = {
        "name": "Comprehensive Health Checkup Package",
        "key_benefits": [
            "Complete blood work analysis",
            "One-on-one consultation with specialists",
            "Personalized health plan",
        ],
        "pricing": {"standard": "$199", "premium": "$349"},
        "promotion_code": "HEALTH2023",
        "promotion_discount": "15% off for returning patients",
    }

    # Generate email content
    print_separator("STANDALONE EMAIL GENERATION DEMO")
    email_content = generate_email_content(
        user_data=user_data,
        campaign_context=campaign_context,
        product_info=product_info,
        brand_voice="caring and professional",
        openai_key=OPENAI_API_KEY,
    )

    # Display the results
    print(f"\nGenerated Email for {user_data['name']}:")
    print(f"Subject: {email_content.get('subject_line', 'No subject')}")
    print(f"Greeting: {email_content.get('greeting', 'Hello')}")
    print(f"Body Content: {email_content.get('body_content', 'No content')}")
    print(f"Call to Action: {email_content.get('call_to_action', 'No CTA')}")
    print(
        f"Personalization Factors: {email_content.get('personalization_factors', [])}"
    )

    return email_content


# Demo function for email template creation from CSV
def demo_email_template_from_csv():
    """
    Demonstrates how to create an email template based on CSV structure.
    """
    # Sample CSV data
    sample_csv = """id,name,address,age,condition,last_visit
1,John Doe,Anytown,40,diabetes,2023-05-15
2,Jane Smith,Othertown,25,asthma,2023-06-20
3,Alice Johnson,Anytown,50,hypertension,2023-04-10
4,Bob Brown,Anytown,60,diabetes,2023-07-05
5,Carol Davis,Sometown,35,allergies,2023-08-12"""

    # Sample campaign context
    campaign_context = [
        "Healthcare follow-up campaign",
        "Focus on preventative care and regular check-ups",
        "Promoting new seasonal services and specialized treatments",
    ]

    # Sample product information
    product_info = {
        "name": "Health Wellness Program",
        "key_benefits": [
            "Personalized health coaching",
            "Regular health screenings",
            "Diet and exercise plans",
        ],
        "pricing": {"monthly": "$49.99", "annual": "$499.99 (save $100)"},
        "promotion_code": "HEALTH2023",
        "promotion_discount": "20% off first 3 months",
    }

    # Generate email template from CSV
    print_separator("EMAIL TEMPLATE FROM CSV DEMO")
    template = create_email_template_from_csv(
        csv_data=sample_csv,
        campaign_context=campaign_context,
        product_info=product_info,
        brand_voice="caring and professional",
        openai_key=OPENAI_API_KEY,
    )

    # Display the results
    print("\nGenerated Email Template:")
    print(
        f"Subject Line Template: {template.get('subject_line_template', 'No template')}"
    )
    print(f"Greeting Template: {template.get('greeting_template', 'No template')}")
    print(
        f"Body Content Template: {template.get('body_content_template', 'No template')}"
    )
    print(
        f"Call to Action Template: {template.get('call_to_action_template', 'No template')}"
    )
    print("\nPersonalization Instructions:")
    for instruction in template.get("personalization_instructions", []):
        print(f"- {instruction}")
    print("\nPlaceholder Mapping:")
    for column, description in template.get("placeholder_mapping", {}).items():
        print(f"- {column}: {description}")

    return template

    # Example usage of filter_csv_with_segments function
    # if __name__ == "__main__":
    # 1. First call find_audience_by_id to get audience data
    print_separator("FINDING AUDIENCE DATA")
    USER_ID_PERSONA = "1246"
    SESSION_TOKEN_PERSONA = ""
    TARGET_ID = "67bc0a7366974b281d5b4b42"

    audience_data = find_audience_by_id(
        user_id=USER_ID_PERSONA,
        SESSION_TOKEN=SESSION_TOKEN_PERSONA,
        target_id=TARGET_ID,
    )
    if audience_data:
        print("Found audience data")
        csv_url = audience_data.get("fileURL")
        response = requests.get(csv_url)
        csv_data = response.text
        file_path = csv_data
    else:
        print("Audience data not found, using sample data instead")
        from io import StringIO

        file_path = StringIO(
            "id,name,address,age,condition\n1,John Doe,Anytown,40,diabetes\n2,Jane Smith,Othertown,25,asthma\n3,Alice Johnson,Anytown,50,hypertension\n4,Bob Brown,Anytown,60,diabetes"
        )

    # 2. Call analyze_csv_columns on the audience data
    print_separator("ANALYZING CSV COLUMNS")
    column_data = analyze_csv_columns(
        file_path, [], is_propensity_data=False, rows="All", seed=42
    )
    print("Column data analysis complete")

    # 3. Process data in batches
    print_separator("PROCESSING DATA IN BATCHES")
    data_batches = process_data_in_batches(column_data)
    print("Data batch processing complete")

    # Create example explanations for columns based on the actual columns in the data
    column_explanations = []
    for col in column_data:
        col_name = col.get("column", "")
        data_type = col.get("data_type", "unknown")
        column_explanations.append(
            f"{col_name}: {col_name.replace('_', ' ').title()} ({data_type})"
        )

    # Use structured_data_tool to generate filter data
    print_separator("CREATING FILTER DATA")
    filter_data = structured_data_tool(
        openai_key=OPENAI_API_KEY,
        audiance_data_dict=data_batches,
        problem_statement="Create targeted segments for a marketing campaign",
        additional_requirements="Focus on creating segments that will be useful for a general audience campaign",
        explanation=column_explanations,
        other_requirements="Ensure segments are meaningful for creating an effective email template for all users",
    )
    print("Filter data generation complete")

    # 4. Filter CSV with segments
    print_separator("FILTERING CSV WITH SEGMENTS")
    filtered_results = filter_csv_with_segments(
        filter_data=filter_data, csv_file_path=file_path, return_dataframe=True
    )
    print("Filtering complete")

    # 5. Create a generalized email template for all users
    print_separator("GENERATING GENERALIZED EMAIL TEMPLATE FOR ALL USERS")

    # Product information to include in email
    product_info = {
        "name": "Customer Engagement Program",
        "key_benefits": [
            "Personalized offers based on preferences",
            "Regular updates on new services",
            "Exclusive members-only content",
        ],
        "promotion_code": "WELCOME2023",
        "promotion_discount": "15% off your next purchase",
    }

    # Get campaign context from filter data
    all_filter_criteria = []
    filter_sets = []

    if "user_upload_filter_columns_result" in filter_data:
        filter_sets = filter_data["user_upload_filter_columns_result"].get("output", [])
    else:
        filter_sets = filter_data.get("output", [])

    # Handle direct nested structure if needed
    if not filter_sets and isinstance(filter_data.get("filter_sets", None), list):
        filter_sets = filter_data.get("filter_sets", [])

    # Extract all filter criteria as context
    for filter_set in filter_sets:
        if (
            isinstance(filter_set, dict)
            and "explanation" in filter_set
            and filter_set["explanation"]
        ):
            all_filter_criteria.append(filter_set["explanation"])
        elif isinstance(filter_set, dict) and "column_names" in filter_set:
            column_names = filter_set.get("column_names", [])
            if column_names:
                criteria_desc = (
                    f"Criteria based on: {', '.join(str(col) for col in column_names)}"
                )
                all_filter_criteria.append(criteria_desc)
        elif isinstance(filter_set, list) and len(filter_set) >= 3:
            # Assuming format [column_names, filter_values, explanation]
            if isinstance(filter_set[2], str) and filter_set[2]:
                all_filter_criteria.append(filter_set[2])
            else:
                column_names = filter_set[0] if isinstance(filter_set[0], list) else []
                if column_names:
                    criteria_desc = f"Criteria based on: {', '.join(str(col) for col in column_names)}"
                    all_filter_criteria.append(criteria_desc)

    # Create a generalized email template for all users
    print("Creating email template for the entire user database...")

    # Use create_email_template_from_csv function to generate a template
    email_template = create_email_template_from_csv(
        csv_data=file_path,
        campaign_context=all_filter_criteria,
        product_info=product_info,
        brand_voice="professional and engaging",
        sample_rows=10,
        openai_key=OPENAI_API_KEY,
    )

    print_separator("FINAL EMAIL TEMPLATE FOR ALL USERS")
    print(
        f"Subject Line Template: {email_template.get('subject_line_template', 'No template')}"
    )
    print(
        f"Greeting Template: {email_template.get('greeting_template', 'No template')}"
    )
    print(
        f"Body Content Template: {email_template.get('body_content_template', 'No template')}"
    )
    print(
        f"Call to Action Template: {email_template.get('call_to_action_template', 'No template')}"
    )

    print("\nPersonalization Instructions:")
    for instruction in email_template.get("personalization_instructions", []):
        print(f"- {instruction}")

    print("\nPlaceholder Mapping:")
    for column, description in email_template.get("placeholder_mapping", {}).items():
        print(f"- {column}: {description}")

    print("\nCSV Columns Available:")
    print(email_template.get("csv_columns", []))
