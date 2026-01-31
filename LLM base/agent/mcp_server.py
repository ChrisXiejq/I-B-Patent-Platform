from typing import Any
import httpx
from fastmcp import FastMCP
from rag.rag_chain import adaptive_rag_answer

# Initialize FastMCP server
mcp = FastMCP("patent")

def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get("event", "Unknown")}
Area: {props.get("areaDesc", "Unknown")}
Severity: {props.get("severity", "Unknown")}
Description: {props.get("description", "No description available")}
Instructions: {props.get("instruction", "No specific instructions provided")}
"""


@mcp.tool()
async def get_identification(user_id: str) -> str:
    """Get user identification based on state code.

    Args:
        user_id: User identifier
    """
    # url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    # data = await make_nws_request(url)
    data = "enterprise"

    return data


@mcp.tool()
async def get_enterprise_interest(patent_no: str) -> str:
    """Get enterprise interest based on patent number.

    Args:
        patent_no: Patent number
    """
    # First get the forecast grid endpoint
    interest = 10

    return "Enterprise interest level: {}".format(interest)

@mcp.tool()
async def get_patent_analysis(patent_no: str) -> str:
    """Get patent analysis based on patent number.

    Args:
        patent_no: Patent number
    """
    # First get the forecast grid endpoint
    analysis = "This patent shows significant innovation in its field."

    return "Patent Analysis: {}".format(analysis)

@mcp.tool()
async def get_rag_patent_info(patent_no: str) -> str:
    """Get rag patent insights based on patent number.

    Args:
        patent_no: Patent number
    """
    # First get the forecast grid endpoint
    insights = adaptive_rag_answer(
        "patent_no {}".format(patent_no),
    )

    return "patent info by rag: {}".format(insights)

def main():
    # Initialize and run the server
    """
    stdio 模式
    """
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()